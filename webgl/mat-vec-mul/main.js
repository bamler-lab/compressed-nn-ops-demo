async function run() {
    const output_elem = document.getElementById("output");
    const gl = document.getElementById("canvas").getContext('webgl2');
    let { vertexBuffer, vertices } = createVertexBuffer(gl);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    const vertexShader = compileShader(gl, "vertex-shader");
    const createPpf = new GlProgram(gl, vertexShader, "createPpf-shader", ["compressedData"], ["cursor"]);
    const matVecMul = new GlProgram(gl, vertexShader, "matVecMul-shader", ["compressedData", "ppf", "inputVec"], ["cursor"]);

    const response = await fetch("compressed_matrices.bin");
    const data = await response.arrayBuffer();

    await new Promise(resolve => setTimeout(resolve, 2000));

    let start = performance.now();
    await new Promise(resolve => setTimeout(resolve, 0));

    const fileHeader = FileHeader.deserialize(data);
    let inputVec = fileHeader.inputVecToGPU(gl);
    let outputVec = createTexture(
        gl, gl.R8I,
        1024, 1, // TODO: we'll need a `maxMatrixDim` field in the file header.
        gl.RED_INTEGER, gl.BYTE,
        null
    );
    let ppf = createTexture(
        gl, gl.R8UI,
        256, 1,
        gl.RED_INTEGER, gl.UNSIGNED_BYTE,
        null
    );

    // console.log(fileHeader);

    const data_u16 = new Uint16Array(data);
    const chunkSize = 1024 * 4096;
    let chunkStart = fileHeader.offsetsAndFileSize[0];
    const compressedData = createTexture(
        gl, gl.R16UI, 1024, 4096, gl.RED_INTEGER, gl.UNSIGNED_SHORT,
        data_u16.subarray(chunkStart, chunkStart + chunkSize) // TODO: deal with case where this is too small.
    );

    for (let i = 0; i != fileHeader.numMatrices; ++i) {
        const begin = fileHeader.offsetsAndFileSize[i];
        const end = fileHeader.offsetsAndFileSize[i + 1];
        if (end - chunkStart > chunkSize) {
            chunkStart = Math.min(begin, data_u16.length - chunkSize);
            gl.bindTexture(gl.TEXTURE_2D, compressedData);
            let chunkData = data_u16.subarray(chunkStart, chunkStart + chunkSize);
            gl.texImage2D(
                gl.TEXTURE_2D, 0, gl.R16UI, 1024, 4096, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT,
                chunkData
            );
        }

        createPpf.run(
            gl,
            [compressedData],
            [ppf],
            256, 1,
            vertexBuffer, vertices,
            [begin - chunkStart]
        );
        // console.log({ i });
        // console.log(downloadTexture(gl, ppf, 256, 1, gl.UNSIGNED_INT));

        matVecMul.run(
            gl,
            [compressedData, ppf, inputVec],
            [outputVec],
            1024, 1,
            vertexBuffer, vertices,
            [begin - chunkStart]
        );
        // console.log(downloadTexture(gl, outputVec, 1024, 1, gl.INT));

        // Swap input and output textures.
        let previousInputVec = inputVec;
        inputVec = outputVec;
        outputVec = previousInputVec;
    }

    console.log(downloadTexture(gl, inputVec, 1024, 1, gl.INT)[0]);

    await new Promise(resolve => setTimeout(resolve, 0));

    let duration = performance.now() - start;
    console.log('duration: ' + duration)

    output_elem.innerHTML = `Duration for decoding 100 matrices of size 1024 x 1024 each and multiplying them to a vector in sequence: ${duration} ms;<br>`
        + `→ i.e., ${(duration * 1e6 / (100 * 1024 * 1024)).toPrecision(3)} ns / matrix element;<br>`
        + `→ i.e., ${(100 * 1024 * 1024 / (1e3 * duration)).toFixed(0)} million decoded matrix elements per second.<br><br>`;
}

/**
 * @param {WebGL2RenderingContext} gl 
 * @param {WebGLTexture} texture 
 * @param {number} width 
 * @param {number} height 
 * @returns {Uint32Array}
 */
function downloadTexture(gl, texture, width, height, type) {
    let output;
    switch (type) {
        case gl.UNSIGNED_INT:
            output = new Uint32Array(4 * width * height);
            break;
        case gl.INT:
            output = new Int32Array(4 * width * height);
            break;
        default:
            throw "unsupported type";
    }

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.readPixels(0, 0, width, height, gl.RGBA_INTEGER, type, output);
    const outputRed = output.filter((_val, i) => i % 4 == 0);
    return outputRed;
}

class FileHeader {
    /**
     * @param {Uint32Array} offsetsAndFileSize
     */
    constructor(offsetsAndFileSize, inputVec) {
        this.offsetsAndFileSize = offsetsAndFileSize;
        this.inputVec = inputVec;
    }

    get numMatrices() {
        return this.offsetsAndFileSize.length - 1;
    }

    /**
     * @param {ArrayBuffer} data
     * @returns {FileHeader} The deserialized FileHeader.
     */
    static deserialize(data) {
        const uint32data = new Uint32Array(data);
        const numMatrices = uint32data[0];
        const offsetsAndFileSize = uint32data.subarray(1, numMatrices + 2);
        const inputDim = uint32data[numMatrices + 2];

        const vectorBegin = 4 * (numMatrices + 3);
        const inputVec = new Int8Array(data, vectorBegin, inputDim);

        return new FileHeader(offsetsAndFileSize, inputVec);
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @returns {WebGLTexture}
     */
    inputVecToGPU(gl) {
        return createTexture(
            gl, gl.R8I,
            this.inputVec.length, 1,
            gl.RED_INTEGER, gl.BYTE,
            this.inputVec
        );
    }
}

/**
 * @param {WebGL2RenderingContext} gl 
 * @param {GLenum} internal_format 
 * @param {number} width 
 * @param {number} height 
 * @param {GLenum} format 
 * @param {GLenum} type 
 * @param {(Uint8Array|Uint16Array|Uint32Array|Float32Array)} pixels 
 * @returns {WebGLTexture}
 */
function createTexture(gl, internal_format, width, height, format, type, pixels) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, internal_format, width, height, 0, format, type, pixels);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

class GlProgram {
    /**
     * Compile and link a WebGL program.
     *
     * @param {WebGL2RenderingContext} gl
     * @param {WebGLShader} vertexShader A (usually boilerplate) compiled vertex shared, as
     *      obtained from `compileShader`.
     * @param {string} fragmentShaderId The `id` of a `script` tag with attribute
     *      `type="x-shader/x-fragment"` that contains the fragment shader code.
     * @param {string[]} inputTextureNames Names of input textures, as they are called in the
     *      shader code.
     * @param {string[]} intUniformNames Names of (scalar) integer uniforms, as they are called in
     *      the shader code.
     */
    constructor(gl, vertexShader, fragmentShaderId, inputTextureNames, intUniformNames) {
        const fragmentShader = compileShader(gl, fragmentShaderId);
        this.program = createProgram(gl, vertexShader, fragmentShader);
        this.textureLocations = inputTextureNames === undefined ? [] : inputTextureNames.map(name => gl.getUniformLocation(this.program, name))
        this.uniformLocations = intUniformNames === undefined ? [] : intUniformNames.map(name => gl.getUniformLocation(this.program, name));
        this.widthLocation = gl.getUniformLocation(this.program, "width");
        this.heightLocation = gl.getUniformLocation(this.program, "height");
        this.vertexPositionLocation = gl.getUniformLocation(this.program, "vertex_position");
    }

    /**
     * Run the WebGL program on some input and output textures.
     *
     * @param {WebGL2RenderingContext} gl 
     * @param {!WebGLTexture[]} inputTextures The input textures, in the same order in which their
     *      names were supplied to the argument `inputTextureNames` of the constructor.
     * @param {!WebGLTexture[]} outputTextures The output textures. They all must have shape `width`
     *      times `height` (see next 2 arguments), and they must be declared in the shader code with
     *      `layout(location=???) out ...`, where `???` is the (zero-based) index into the array
     *      `outputTextures`.
     * @param {number} width The width of all output textures.
     * @param {number} height The height of all output textures.
     * @param {WebGLBuffer} vertexBuffer A (usually dummy) vertex buffer, e.g., as obtained from
     *      `createVertexBuffer`.
     * @param {Float32Array} vertices A (usually dummy) vertex buffer, e.g., as obtained from
     *      `createVertexBuffer`.
     * @param {number[]} intUniforms Array of (scalar) integer uniforms. May be omitted if the
     *      `GlProgram` was constructed without any `intUniformNames`.
     */
    run(gl, inputTextures, outputTextures, width, height, vertexBuffer, vertices, intUniforms) {
        gl.viewport(0, 0, width, height);
        gl.useProgram(this.program);

        gl.uniform1i(this.widthLocation, width);
        gl.uniform1i(this.heightLocation, height);
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
        gl.vertexAttribPointer(this.vertexPositionLocation, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.vertexPositionLocation);

        for (let i = 0; i != this.uniformLocations.length; ++i) {
            gl.uniform1i(this.uniformLocations[i], intUniforms[i]);
        }

        // WebGL is guaranteed to have at least 8 texture units.
        const gl_textures = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3, gl.TEXTURE4, gl.TEXTURE5, gl.TEXTURE6, gl.TEXTURE7];
        for (let i = 0; i != inputTextures.length; ++i) {
            gl.uniform1i(this.textureLocations[i], i);
            gl.activeTexture(gl_textures[i]);
            gl.bindTexture(gl.TEXTURE_2D, inputTextures[i]);
        }

        // Not sure how many color attachments we can rely on.
        const gl_color_attachments = [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3];
        for (let i = 0; i != outputTextures.length; ++i) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl_color_attachments[i], gl.TEXTURE_2D, outputTextures[i], 0);
            // TODO: maybe use gl.DRAW_FRAMEBUFFER
        }

        gl.drawBuffers(gl_color_attachments.slice(0, outputTextures.length));
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        for (let i = 0; i != outputTextures.length; ++i) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl_color_attachments[i], gl.TEXTURE_2D, null, 0);
        }
    }
}

/**
 * Create dummy vertex buffer, and a list of vertices that define two triangles that cover the
 * rectangle `[-1, 1] x [-1, 1]`.
 * @param {WebGL2RenderingContext} gl 
 * @returns {{vertices: Float32Array, vertexBuffer: WebGLBuffer}}`
 */
function createVertexBuffer(gl) {
    let vertices = new Float32Array([
        -1., 1., 1., 1., 1., -1., // Triangle 1
        -1., 1., 1., -1., -1., -1. // Triangle 2 
    ]);
    let vertexBuffer = gl.createBuffer();
    return { vertices, vertexBuffer };
}

/**
 * Creates and compiles a vertex or fragment shader.
 *
 * @param {!WebGLRenderingContext} gl The WebGL Context.
 * @param {string} scriptId The id of the script tag. The tag must have a `type` attribute of either
 *      `"x-shader/x-vertex"` or `"x-shader/x-fragment"`.
 * @return {!WebGLShader} The compiled shader.
 */
function compileShader(gl, scriptId) {
    const shaderScript = document.getElementById(scriptId);
    if (!shaderScript) {
        throw "unknown script element: " + scriptId;
    }

    let shaderType;
    switch (shaderScript.type) {
        case "x-shader/x-vertex":
            shaderType = gl.VERTEX_SHADER;
            break;
        case "x-shader/x-fragment":
            shaderType = gl.FRAGMENT_SHADER;
            break;
        default:
            throw "shader type not set";
    }

    const shader = gl.createShader(shaderType);
    gl.shaderSource(shader, shaderScript.text);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw "could not compile shader: " + gl.getShaderInfoLog(shader);
    }

    return shader;
}

/**
 * Creates a program from a vertex and fragment shader.
 *
 * @param {!WebGLRenderingContext} gl The WebGL context.
 * @param {!WebGLShader} vertexShader A (compiled) vertex shader.
 * @param {!WebGLShader} fragmentShader A (compiled) fragment shader.
 * @return {!WebGLProgram} A program.
 */
function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw "program failed to link: " + gl.getProgramInfoLog(program);
    }

    return program;
};
