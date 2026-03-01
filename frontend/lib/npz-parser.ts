import { unzipSync } from 'fflate'

/**
 * Parse NPZ (compressed NumPy archive) file in the browser.
 * NPZ files are ZIP archives containing .npy files.
 */
export async function parseNPZ(
  buffer: ArrayBuffer
): Promise<Record<string, Float32Array | Uint32Array>> {
  const uint8Array = new Uint8Array(buffer)
  
  // Decompress the ZIP archive
  const unzipped = unzipSync(uint8Array)
  
  const result: Record<string, Float32Array | Uint32Array> = {}
  
  // Process each file in the archive
  for (const [filename, fileData] of Object.entries(unzipped)) {
    if (!filename.endsWith('.npy')) {
      continue
    }
    
    // Remove .npy extension for the key
    const key = filename.replace(/\.npy$/, '')
    
    // Parse NPY file
    const array = parseNPY(fileData)
    result[key] = array
  }
  
  return result
}

/**
 * Parse a single NPY file.
 * NPY format: magic bytes (6) + version (2) + header length (2/4) + header (JSON dict) + data
 */
function parseNPY(data: Uint8Array): Float32Array | Uint32Array {
  let offset = 0
  
  // Check magic bytes: \x93NUMPY
  const magic = data.slice(0, 6)
  const expectedMagic = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59])
  for (let i = 0; i < 6; i++) {
    if (magic[i] !== expectedMagic[i]) {
      throw new Error('Invalid NPY file: magic bytes mismatch')
    }
  }
  offset += 6
  
  // Version (major, minor)
  const majorVersion = data[offset]
  const minorVersion = data[offset + 1]
  offset += 2
  
  // Header length
  let headerLength: number
  if (majorVersion === 1) {
    // Version 1.0: 2-byte little-endian header length
    headerLength = data[offset] | (data[offset + 1] << 8)
    offset += 2
  } else if (majorVersion === 2 || majorVersion === 3) {
    // Version 2.0/3.0: 4-byte little-endian header length
    headerLength = data[offset] | (data[offset + 1] << 8) | 
                   (data[offset + 2] << 16) | (data[offset + 3] << 24)
    offset += 4
  } else {
    throw new Error(`Unsupported NPY version: ${majorVersion}.${minorVersion}`)
  }
  
  // Parse header (Python dict as string)
  const headerBytes = data.slice(offset, offset + headerLength)
  const headerStr = new TextDecoder('utf-8').decode(headerBytes)
  offset += headerLength
  
  // Extract dtype and shape from header
  const header = parseNPYHeader(headerStr)
  
  // Read data based on dtype
  const dataView = new DataView(data.buffer, data.byteOffset + offset)
  const totalElements = header.shape.reduce((a, b) => a * b, 1)
  
  let typedArray: Float32Array | Uint32Array
  
  if (header.dtype === 'float32' || header.dtype === '<f4' || header.dtype === 'f4') {
    typedArray = new Float32Array(totalElements)
    for (let i = 0; i < totalElements; i++) {
      typedArray[i] = dataView.getFloat32(i * 4, true) // little-endian
    }
  } else if (header.dtype === 'int32' || header.dtype === '<i4' || header.dtype === 'i4') {
    // Convert Int32 to Uint32 for WebGL compatibility
    typedArray = new Uint32Array(totalElements)
    for (let i = 0; i < totalElements; i++) {
      typedArray[i] = dataView.getInt32(i * 4, true) // little-endian
    }
  } else if (header.dtype === 'float64' || header.dtype === '<f8' || header.dtype === 'f8') {
    // Convert float64 to float32
    const tempArray = new Float64Array(totalElements)
    for (let i = 0; i < totalElements; i++) {
      tempArray[i] = dataView.getFloat64(i * 8, true)
    }
    typedArray = new Float32Array(tempArray)
  } else {
    throw new Error(`Unsupported dtype: ${header.dtype}`)
  }
  
  return typedArray
}

/**
 * Parse NPY header string (Python dict format).
 * Example: "{'descr': '<f4', 'fortran_order': False, 'shape': (100, 3), }"
 */
function parseNPYHeader(headerStr: string): { dtype: string; shape: number[] } {
  // Extract dtype (descr field)
  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/)
  if (!descrMatch) {
    throw new Error('Could not parse dtype from NPY header')
  }
  let dtype = descrMatch[1]
  
  // Normalize dtype
  if (dtype.startsWith('<') || dtype.startsWith('>') || dtype.startsWith('|')) {
    dtype = dtype.substring(1) // Remove endianness marker
  }
  
  // Extract shape
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/)
  if (!shapeMatch) {
    throw new Error('Could not parse shape from NPY header')
  }
  
  const shapeStr = shapeMatch[1].trim()
  const shape: number[] = []
  
  if (shapeStr) {
    const parts = shapeStr.split(',').map(s => s.trim()).filter(s => s)
    for (const part of parts) {
      const num = parseInt(part, 10)
      if (isNaN(num)) {
        throw new Error(`Invalid shape value: ${part}`)
      }
      shape.push(num)
    }
  }
  
  // Handle scalar case (empty shape means 0D array)
  if (shape.length === 0) {
    shape.push(1)
  }
  
  return { dtype, shape }
}
