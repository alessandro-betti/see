function getDataFromNumPyArray(buf) {
    var magic = String.fromCharCode.apply(null, new Uint8Array(buf.slice(0, 6)));
    if (magic.slice(1, 6) != 'NUMPY') {
        throw new Error('Unknown file type, expected: NUMPY');
    }

    var version = new Uint8Array(buf.slice(6, 8));
    var view = new DataView(buf.slice(8, 10));
    var headerLength = view.getUint8(0);
    headerLength |= view.getUint8(1) << 8;
    var headerStr = String.fromCharCode.apply(null, new Uint8Array(buf.slice(10, 10 + headerLength)));
    var offsetBytes = 10 + headerLength;
    var info;
    eval("info = " + headerStr.toLowerCase().replace('(', '[').replace('),', ']'));

    var h = info.shape[0];
    var w = info.shape[1];
    var c = info.shape[2];

    var data;
    if (info.descr === "|u1") {
        data = new Uint8Array(buf, offsetBytes);
    } else if (info.descr === "|i1") {
        data = new Int8Array(buf, offsetBytes);
    } else if (info.descr === "<u2") {
        data = new Uint16Array(buf, offsetBytes);
    } else if (info.descr === "<i2") {
        data = new Int16Array(buf, offsetBytes);
    } else if (info.descr === "<u4") {
        data = new Uint32Array(buf, offsetBytes);
    } else if (info.descr === "<i4") {
        data = new Int32Array(buf, offsetBytes);
    } else if (info.descr === "<f4") {
        data = new Float32Array(buf, offsetBytes);
    } else if (info.descr === "<f8") {
        data = new Float64Array(buf, offsetBytes);
    } else {
        throw new Error('Unknown numeric dtype!')
    }

    return {data: data, w: w, h: h, c: c}
}

function copyAndGoGrayscale(dest_pixels, source_pixels) {
    var g;
    for (var i = 0; i < source_pixels.length; i = i + 4) {
        g = source_pixels[i] * .3 + source_pixels[i + 1] * .59 + source_pixels[i + 2] * .11;
        dest_pixels[i + 0] = g; // R
        dest_pixels[i + 1] = g; // G
        dest_pixels[i + 2] = g; // B
        dest_pixels[i + 3] = source_pixels[i + 3]; // A
    }
}

function goBlack(pixels) {
    for (var i = 0; i < pixels.length; i = i + 4) {
        pixels[i + 0] = 0; // R
        pixels[i + 1] = 0; // G
        pixels[i + 2] = 0; // B
        pixels[i + 3] = 255; // A
    }
}

function goWhite(pixels) {
    for (var i = 0; i < pixels.length; i = i + 4) {
        pixels[i + 0] = 255; // R
        pixels[i + 1] = 255; // G
        pixels[i + 2] = 255; // B
        pixels[i + 3] = 255; // A
    }
}

function goRed(pixels) {
    for (var i = 0; i < pixels.length; i = i + 4) {
        pixels[i + 0] = 255; // R
        pixels[i + 1] = 0; // G
        pixels[i + 2] = 0; // B
        pixels[i + 3] = 255; // A
    }
}

function copyPixels(dest_pixels, source_pixels) {
    for (var i = 0; i < dest_pixels.length; i = i + 4) {
        dest_pixels[i + 0] = source_pixels[i + 0]; // R
        dest_pixels[i + 1] = source_pixels[i + 1]; // G
        dest_pixels[i + 2] = source_pixels[i + 2]; // B
        dest_pixels[i + 3] = source_pixels[i + 3]; // A
    }
}

function scalePix(pixels, rho) {
    for (var i = 0; i < pixels.length; i = i + 4) {
        pixels[i + 0] = rho * pixels[i + 0]; // R
        pixels[i + 1] = rho * pixels[i + 1]; // G
        pixels[i + 2] = rho * pixels[i + 2]; // B
    }
}

function hsvToRgb(h, s, v) {
    var r, g, b;

    var i = Math.floor(h * 6);
    var f = h * 6 - i;
    var p = v * (1 - s);
    var q = v * (1 - f * s);
    var t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0:
            r = v, g = t, b = p;
            break;
        case 1:
            r = q, g = v, b = p;
            break;
        case 2:
            r = p, g = v, b = t;
            break;
        case 3:
            r = p, g = q, b = v;
            break;
        case 4:
            r = t, g = p, b = v;
            break;
        case 5:
            r = v, g = p, b = q;
            break;
    }

    return [r * 255, g * 255, b * 255];
}