async function run() {
    const gl = document.getElementById("canvas").getContext('webgl2');

    const response = await fetch("compressed_matrices.bin");
    const buf = await response.arrayBuffer();
    const data = new Reader(buf);

    const fileHeader = FileHeader.deserialize(data);
    const inputVec = fileHeader.inputVecToGPU(gl);

    console.log(fileHeader);
}

class Reader {
    /**
     * @param {ArrayBuffer} data
     */
    constructor(data) {
        this.data = new Uint16Array(data);
        this.cursor = 0;
    }
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
     * @param {Reader} reader 
     * @returns {FileHeader}
     */
    static deserialize(reader) {
        const uint32data = new Uint32Array(reader.data.buffer, 2 * reader.cursor);
        const numMatrices = uint32data[0];
        const offsetsAndFileSize = uint32data.subarray(1, numMatrices + 2);
        const inputDim = uint32data[numMatrices + 2];
        reader.cursor += 2 * (numMatrices + 3);

        const inputVec = new Uint8Array(reader.data.buffer, 2 * reader.cursor, inputDim);
        reader.cursor += Math.floor((inputDim + 1) / 2);

        return new FileHeader(offsetsAndFileSize, inputVec);
    }

    /**
     * @param {WebGL2RenderingContext} gl
     * @returns {WebGLTexture}
     */
    inputVecToGPU(gl) {
        createTexture(
            gl, gl.R8UI,
            this.inputVec.length, 1,
            gl.RED_INTEGER, gl.UNSIGNED_BYTE,
            this.inputVec
        )
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
    constructor(gl, vertexShader, fragmentShaderId, textureNames, uniformNames) {
        const fragmentShader = compileShader(gl, fragmentShaderId);
        this.program = createProgram(gl, vertexShader, fragmentShader);
        this.textureLocations = textureNames === undefined ? [] : textureNames.map(name => gl.getUniformLocation(this.program, name))
        this.uniformLocations = uniformNames === undefined ? [] : uniformNames.map(name => gl.getUniformLocation(this.program, name));
        this.widthLocation = gl.getUniformLocation(this.program, "width");
        this.heightLocation = gl.getUniformLocation(this.program, "height");
        this.vertexPositionLocation = gl.getUniformLocation(this.program, "vertex_position");
    }

    run(gl, inputTextures, outputTextures, width, height, vertexBuffer, vertices, uniforms) {
        gl.viewport(0, 0, width, height);
        gl.useProgram(this.program);

        gl.uniform1i(this.widthLocation, width);
        gl.uniform1i(this.heightLocation, height);
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
        gl.vertexAttribPointer(this.vertexPositionLocation, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.vertexPositionLocation);

        for (let i = 0; i != this.uniformLocations.length; ++i) {
            gl.uniform1i(this.uniformLocations[i], uniforms[i]);
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
