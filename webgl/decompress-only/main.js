async function run() {
    const canvas = document.getElementById("canvas");
    const output_elem = document.getElementById("output");
    const gl = getGl(canvas);
    const vertexShader = compileShader(gl, "vertex-shader");

    const response = await fetch("100-compressed-matrices.bin");
    const buf = await response.arrayBuffer();
    const data = new Uint16Array(buf);

    let coderHeadsIn = createTexture(gl, gl.RG16UI, 1024, 1024, gl.RG_INTEGER, gl.UNSIGNED_SHORT, data.subarray(0, 1024 * 1024 * 2));
    let coderHeadsOut = createTexture(gl, gl.RG16UI, 1024, 1024, gl.RG_INTEGER, gl.UNSIGNED_SHORT, null);
    const lookupTable = createTexture(gl, gl.R8UI, 4096, 1, gl.RED_INTEGER, gl.UNSIGNED_BYTE, null);

    const coderOffsets1 = createTexture(gl, gl.R8UI, 1024, 1024, gl.RED_INTEGER, gl.UNSIGNED_BYTE, new Uint8Array(1024 * 1024));
    const coderOffsets2 = createTexture(gl, gl.R8UI, 256, 1024, gl.RED_INTEGER, gl.UNSIGNED_BYTE, new Uint8Array(256 * 1024));
    const coderOffsets3 = createTexture(gl, gl.R16UI, 32, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, new Uint16Array(32 * 1024));
    const coderOffsets4 = createTexture(gl, gl.R16UI, 1, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, new Uint16Array(2 * 1024));
    const coderOffsets5 = createTexture(gl, gl.R32UI, 1, 32, gl.RED_INTEGER, gl.UNSIGNED_INT, new Uint32Array(1 * 32));

    let { vertexBuffer, vertices } = createVertexBuffer(gl);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    const createLookup = new GlProgram(gl, vertexShader, "create-lookup-shader", ["compressedData"], ["cursor"]);
    const decode = new GlProgram(
        gl, vertexShader, "decode-shader",
        ["coderHeads", "compressedData", "lookupTable", "coderOffsets1", "coderOffsets2", "coderOffsets3", "coderOffsets4", "coderOffsets5"],
        ["cursor"]
    );

    const sumOffsets0 = new GlProgram(gl, vertexShader, "sumOffsets0-shader", ["coderHeads"]);
    const sumOffsets1 = new GlProgram(gl, vertexShader, "sumOffsets1-shader", ["coderOffsets1"]);
    const sumOffsets2 = new GlProgram(gl, vertexShader, "sumOffsets2-shader", ["coderOffsets2"]);
    const sumOffsets3 = new GlProgram(gl, vertexShader, "sumOffsets3-shader", ["coderOffsets3"]);
    const sumOffsets4 = new GlProgram(gl, vertexShader, "sumOffsets4-shader", ["coderOffsets4"]);

    await new Promise(resolve => setTimeout(resolve, 2000));

    let start = performance.now();
    await new Promise(resolve => setTimeout(resolve, 0));

    const serializedSizes = data.subarray(1024 * 1024 * 2, 1024 * 1024 * 2 + 100 * 2);
    const chunkSize = 1024 * 1024;
    let chunkStart = 1024 * 1024 * 2 + 100 * 2;
    const compressedData = createTexture(gl, gl.R16UI, 1024, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, data.subarray(chunkStart, chunkStart + chunkSize));
    let cursor = 0;

    matrices = [];

    for (let i = 0; i != 100; ++i) {
        const matrix = createTexture(gl, gl.R16UI, 1024, 1024, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);
        const currentSize = serializedSizes[2 * i] | (serializedSizes[2 * i + 1] << 16);

        if (cursor + currentSize >= chunkSize) {
            chunkStart += cursor;
            cursor = 0;
            gl.bindTexture(gl.TEXTURE_2D, compressedData);
            let dat = data.subarray(chunkStart, chunkStart + chunkSize);
            if (dat.length !== chunkSize) {
                // TODO: find a more elegant way to do this.
                const tmp = dat;
                dat = new Uint16Array(chunkSize);
                dat.subarray(0, tmp.length).set(tmp);
            }
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, 1024, 1024, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, dat);
        }

        createLookup.run(gl, [compressedData], [lookupTable], 4096, 1, vertexBuffer, vertices, [cursor]);
        decode.run(
            gl,
            [coderHeadsIn, compressedData, lookupTable, coderOffsets1, coderOffsets2, coderOffsets3, coderOffsets4, coderOffsets5],
            [matrix, coderHeadsOut],
            1024, 1024, vertexBuffer, vertices,
            [cursor]
        );
        cursor += currentSize;

        sumOffsets0.run(gl, [coderHeadsOut], [coderOffsets1], 1024, 1024, vertexBuffer, vertices);
        sumOffsets1.run(gl, [coderOffsets1], [coderOffsets2], 256, 1024, vertexBuffer, vertices);
        sumOffsets2.run(gl, [coderOffsets2], [coderOffsets3], 32, 1024, vertexBuffer, vertices);
        sumOffsets3.run(gl, [coderOffsets3], [coderOffsets4], 1, 1024, vertexBuffer, vertices);
        sumOffsets4.run(gl, [coderOffsets4], [coderOffsets5], 1, 32, vertexBuffer, vertices);

        let swapTmp = coderHeadsIn;
        coderHeadsIn = coderHeadsOut;
        coderHeadsOut = swapTmp;

        matrices.push(matrix);

        // gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, coderOffsets5, 0);
        // const rawBuffer = new ArrayBuffer(16);
        // const buffer = new Uint32Array(rawBuffer);
        // gl.readPixels(0, 31, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, buffer);
        // nextSize = buffer[0];
        // console.log({ nextSize });
    }

    // gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, coderOffsets5, 0);
    // let rawBuffer = new ArrayBuffer(1 * 32 * 4 * 4);
    // let output = new Uint32Array(rawBuffer);
    // gl.readPixels(0, 0, 1, 32, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output);
    // console.log(output.filter((_val, i) => i % 4 == 0));
    // console.log({ sum: output.filter((_val, i) => i % 4 == 0).reduce((s, x) => s + x, 0) });

    // Read from the last decoded matrix to ensure that decoding has actually happened
    // at this point and isn't still scheduled to run in the background (it suffices to
    // read only from the last matrix; the last matrix can only be decoded if all
    // previous matrices were also decoded).
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, matrices[99], 0);
    let rawBuffer = new ArrayBuffer(1 * 1 * 4 * 4);
    let output = new Uint32Array(rawBuffer);
    gl.readPixels(567, 234, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output);
    console.log('matrix99[234, 567]: ' + output[0])

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, matrices[0], 0);
    rawBuffer = new ArrayBuffer(1 * 1 * 4 * 4);
    output = new Uint32Array(rawBuffer);
    gl.readPixels(0, 0, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output);
    console.log('matrix0[0, 0]: ' + output[0])

    await new Promise(resolve => setTimeout(resolve, 0));
    let end = performance.now();
    console.log('duration: ' + (end - start))
    output_elem.innerHTML = `Duration for decoding 100 matrices of size 1024 x 1024 each: ${end - start} ms;<br>`
        + `→ i.e., ${((end - start) * 1e6 / (100 * 1024 * 1024)).toPrecision(3)} ns / matrix element;<br>`
        + `→ i.e., ${(100 * 1024 * 1024 / (1e3 * (end - start))).toFixed(0)} million decoded matrix elements per second.<br><br>`;

    for (let i = 0; i != 100; ++i) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, matrices[i], 0);
        rawBuffer = new ArrayBuffer(1024 * 1024 * 4 * 4);
        output = new Uint32Array(rawBuffer);
        gl.readPixels(0, 0, 1024, 1024, gl.RGBA_INTEGER, gl.UNSIGNED_INT, output);
        const flat_matrix = output.filter((_val, i) => i % 4 == 0);

        // Checksum adapted from rust's FxHash, but restricted to 26 bit hashes so that the
        // multiplication doesn't exceed the range of exactly representable integers in JavaScript.
        let checksum = 0;
        for (let value of flat_matrix) {
            checksum = (((checksum & 0x001f_ffff) << 5) | (checksum >> 21)) >>> 0; // rotate
            checksum = ((checksum ^ value) >>> 0) * 0x0322_0a95;
            checksum = (checksum & 0x03ff_ffff) >>> 0 // truncate to 26 bit
        }
        output_elem.innerHTML += `checksum of matrix ${i}: ${checksum}<br>`;
        await new Promise(resolve => setTimeout(resolve, 0));
    }
}

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

function createOffsetTexture(gl) {
    const data = new Uint8Array(Array.from({ length: 1024 * 1024 }, (_, i) => i % 100));
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8UI, 1024, 1024, 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE, data);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

function createEmptySumsTexture(gl, width, height) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, width, height, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}


function createEmptyLookupTableTexture(gl) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Upload the (still empty) texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, 1024, 1024, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

async function createInputDataTexture(gl) {
    const response = await fetch("mock-matrix.bin");
    const buf = await response.arrayBuffer();
    const data = new Uint16Array(buf);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, 1024, 328, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, data);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
}

function createVectorTexture(gl) {
    const data = new Int16Array(Array.from({ length: 1024 }, (_, i) => i));
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Upload the texture to the GPU:
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16I, 1024, 1, 0, gl.RED_INTEGER, gl.SHORT, data);

    // can't filter integer textures
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

    return texture;
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
 * Obtains the GL context.
 *
 * @param {canvas} HTMLCanvasElement The canvas element.
 * @return {!WebGL2RenderingContext} The WebGL rendering context.
 */
function getGl(canvas) {
    return canvas.getContext('webgl2');
}

/**
 * Creates and compiles a shader.
 *
 * @param {!WebGLRenderingContext} gl The WebGL Context.
 * @param {string} scriptId The id of the script tag.
 * @return {!WebGLShader} The shader.
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
 * Creates a program from 2 shaders.
 *
 * @param {!WebGLRenderingContext} gl The WebGL context.
 * @param {!WebGLShader} vertexShader A vertex shader.
 * @param {!WebGLShader} fragmentShader A fragment shader.
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
