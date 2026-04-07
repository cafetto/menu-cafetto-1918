import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { uniform, vec2, vec3, vec4, float, int, hash, screenUV, time, clamp, pass, mrt, output, transformedNormalView, Fn, Loop, If, dot, fract, floor, abs, sin, cos, mix, step, max, min, normalize, length, renderOutput } from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import { ao } from 'three/examples/jsm/tsl/display/GTAONode.js';
import { dof } from 'three/examples/jsm/tsl/display/DepthOfFieldNode.js';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';

// ─── Scene ───
const scene = new THREE.Scene();
scene.backgroundBlurriness = 0.3;

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(7, 2, 6.5);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.50;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.VSMShadowMap;
const root = document.getElementById('root') ?? document.body;
root.appendChild(renderer.domElement);
await renderer.init();

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 0, 0);
controls.minDistance = 5;
controls.maxDistance = 40;
controls.maxPolarAngle = Math.PI * 0.48;

// ─── Procedural textures ───
function createCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return { canvas: c, ctx: c.getContext('2d') };
}

function seededRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// ─── High-quality procedural wood PBR texture generators ───

// Simplex-like noise for wood grain
function woodNoise2D(x, y, seed) {
  const rng = (a, b) => {
    let h = (a * 12.9898 + b * 78.233 + seed * 43.21) * 43758.5453;
    return h - Math.floor(h);
  };
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const n00 = rng(ix, iy), n10 = rng(ix + 1, iy);
  const n01 = rng(ix, iy + 1), n11 = rng(ix + 1, iy + 1);
  return n00 * (1 - sx) * (1 - sy) + n10 * sx * (1 - sy) + n01 * (1 - sx) * sy + n11 * sx * sy;
}

function fbmWood(x, y, seed, octaves = 4) {
  let val = 0, amp = 0.5, freq = 1;
  for (let i = 0; i < octaves; i++) {
    val += amp * woodNoise2D(x * freq, y * freq, seed + i * 71.3);
    amp *= 0.5; freq *= 2.1;
  }
  return val;
}

// Flat normal map (default neutral normal rgb(128,128,255))
function generateNormalMap() {
  const size = 4;
  const { canvas, ctx } = createCanvas(size, size);
  ctx.fillStyle = 'rgb(128,128,255)';
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

function generateWoodColorMap(baseColor, seed = 42) {
  const size = 256;
  const { canvas, ctx } = createCanvas(size, size);
  const imgData = ctx.createImageData(size, size);
  const d = imgData.data;
  const base = new THREE.Color(baseColor);
  const rng = seededRandom(seed);

  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const u = px / size, v = py / size;
      const ringFreq = 18 + rng() * 4;
      const warp = fbmWood(u * 3, v * 3, seed, 3) * 1.2;
      const ring = Math.sin((v * ringFreq + warp) * Math.PI * 2) * 0.5 + 0.5;
      const grain = fbmWood(u * 40, v * 6, seed + 100, 3) * 0.15;
      const knot = Math.max(0, 1.0 - Math.sqrt((u - 0.3) * (u - 0.3) * 25 + (v - 0.6) * (v - 0.6) * 25)) * 0.15 * rng();
      const bright = 0.7 + ring * 0.25 + grain - knot;
      const darkVar = fbmWood(u * 8, v * 2, seed + 200, 2) * 0.1;
      const r = Math.min(1, Math.max(0, base.r * bright - darkVar + rng() * 0.015));
      const g = Math.min(1, Math.max(0, base.g * bright - darkVar * 0.8 + rng() * 0.01));
      const b = Math.min(1, Math.max(0, base.b * bright - darkVar * 0.5 + rng() * 0.01));
      const i = (py * size + px) * 4;
      d[i] = r * 255; d[i + 1] = g * 255; d[i + 2] = b * 255; d[i + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function generateWoodNormalMap(seed = 42) {
  const size = 256;
  const { canvas, ctx } = createCanvas(size, size);
  const imgData = ctx.createImageData(size, size);
  const d = imgData.data;
  const heights = new Float32Array(size * size);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const u = px / size, v = py / size;
      const ringFreq = 18;
      const warp = fbmWood(u * 3, v * 3, seed, 3) * 1.2;
      const ring = Math.sin((v * ringFreq + warp) * Math.PI * 2) * 0.5 + 0.5;
      const grain = fbmWood(u * 40, v * 6, seed + 100, 3) * 0.3;
      const fine = fbmWood(u * 80, v * 12, seed + 300, 2) * 0.1;
      heights[py * size + px] = ring * 0.6 + grain + fine;
    }
  }
  const strength = 2.5;
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const l = heights[py * size + ((px - 1 + size) % size)];
      const r = heights[py * size + ((px + 1) % size)];
      const t = heights[((py - 1 + size) % size) * size + px];
      const bv = heights[((py + 1) % size) * size + px];
      let nx = (l - r) * strength;
      let ny = (t - bv) * strength;
      let nz = 1.0;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      nx /= len; ny /= len; nz /= len;
      const i = (py * size + px) * 4;
      d[i] = (nx * 0.5 + 0.5) * 255;
      d[i + 1] = (ny * 0.5 + 0.5) * 255;
      d[i + 2] = (nz * 0.5 + 0.5) * 255;
      d[i + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  return tex;
}

function generateWoodBumpMap(seed = 42) {
  const size = 256;
  const { canvas, ctx } = createCanvas(size, size);
  const imgData = ctx.createImageData(size, size);
  const d = imgData.data;
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const u = px / size, v = py / size;
      const ringFreq = 18;
      const warp = fbmWood(u * 3, v * 3, seed, 3) * 1.2;
      const ring = Math.sin((v * ringFreq + warp) * Math.PI * 2) * 0.5 + 0.5;
      const grain = fbmWood(u * 40, v * 6, seed + 100, 3) * 0.25;
      const fine = fbmWood(u * 80, v * 12, seed + 300, 2) * 0.15;
      const h = Math.min(1, Math.max(0, ring * 0.5 + grain + fine + 0.2));
      const i = (py * size + px) * 4;
      d[i] = d[i + 1] = d[i + 2] = h * 255; d[i + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  return tex;
}

function generateWoodRoughnessMap(seed = 42) {
  const size = 256;
  const { canvas, ctx } = createCanvas(size, size);
  const imgData = ctx.createImageData(size, size);
  const d = imgData.data;
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const u = px / size, v = py / size;
      const grain = fbmWood(u * 30, v * 5, seed + 400, 3);
      const fine = fbmWood(u * 60, v * 10, seed + 500, 2) * 0.15;
      const roughness = 0.35 + grain * 0.2 + fine;
      const clamped = Math.min(1, Math.max(0, roughness));
      const i = (py * size + px) * 4;
      d[i] = d[i + 1] = d[i + 2] = clamped * 255; d[i + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  return tex;
}

// Shared PBR wood maps (generated once, reused by all wood materials)
const woodNormalTex = generateWoodNormalMap(42);
const woodBumpTex = generateWoodBumpMap(42);
const woodRoughnessTex = generateWoodRoughnessMap(42);

function generateWoodGrainNormal() {
  return woodNormalTex;
}

function generateClearcoatNormalMap() {
  const size = 128;
  const { canvas, ctx } = createCanvas(size, size);
  const imgData = ctx.createImageData(size, size);
  const d = imgData.data;
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const u = px / size, v = py / size;
      const n = fbmWood(u * 60, v * 60, 777, 2) * 0.08;
      const i = (py * size + px) * 4;
      d[i] = (n * 0.5 + 0.5) * 255;
      d[i + 1] = (n * 0.5 + 0.5) * 255;
      d[i + 2] = 255;
      d[i + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

const clearcoatNormalTex = generateClearcoatNormalMap();

// ─── Environment ───
const rgbeLoader = new RGBELoader();
const pmremGenerator = new THREE.PMREMGenerator(renderer);

function generateFallbackEnvMap() {
  const size = 128;
  const { canvas, ctx } = createCanvas(size * 4, size * 2);
  const grad = ctx.createLinearGradient(0, 0, 0, size * 2);
  grad.addColorStop(0, '#87CEEB');
  grad.addColorStop(0.4, '#B0D4E8');
  grad.addColorStop(0.5, '#E8DCC8');
  grad.addColorStop(1, '#8B7355');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size * 4, size * 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

const fallbackEnv = generateFallbackEnvMap();
scene.environment = fallbackEnv;
scene.background = fallbackEnv;

// Procedural night HDRI map — detailed sky with milky way, nebulae, many stars
function generateNightEnvMap() {
  const size = 256;
  const w = size * 4, h = size * 2;
  const { canvas, ctx } = createCanvas(w, h);

  // Base sky gradient — slightly brighter for ambient contribution
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, '#08081a');
  grad.addColorStop(0.10, '#0c0e28');
  grad.addColorStop(0.25, '#101438');
  grad.addColorStop(0.40, '#141a45');
  grad.addColorStop(0.48, '#161e4a');
  grad.addColorStop(0.52, '#0e1228');
  grad.addColorStop(0.65, '#0a0e1c');
  grad.addColorStop(0.80, '#080a14');
  grad.addColorStop(1, '#04060c');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  const rng = seededRandom(9999);

  // Milky Way band — diagonal hazy glow across the upper sky
  ctx.save();
  ctx.translate(w * 0.5, h * 0.22);
  ctx.rotate(-0.25);
  const mwGrad = ctx.createLinearGradient(-w * 0.5, 0, w * 0.5, 0);
  mwGrad.addColorStop(0, 'rgba(20,22,50,0)');
  mwGrad.addColorStop(0.2, 'rgba(40,42,80,0.12)');
  mwGrad.addColorStop(0.35, 'rgba(60,58,100,0.18)');
  mwGrad.addColorStop(0.5, 'rgba(70,65,110,0.22)');
  mwGrad.addColorStop(0.65, 'rgba(55,52,90,0.16)');
  mwGrad.addColorStop(0.8, 'rgba(35,32,65,0.10)');
  mwGrad.addColorStop(1, 'rgba(15,14,35,0)');
  ctx.fillStyle = mwGrad;
  ctx.fillRect(-w * 0.55, -h * 0.08, w * 1.1, h * 0.16);
  ctx.restore();

  // Dense star cluster in milky way core
  for (let i = 0; i < 400; i++) {
    const angle = (rng() - 0.5) * 0.6;
    const along = (rng() - 0.5) * w * 0.8;
    const sx = w * 0.5 + along * Math.cos(-0.25) + (rng() - 0.5) * h * 0.08;
    const sy = h * 0.22 + along * Math.sin(-0.25) + (rng() - 0.5) * h * 0.06;
    if (sy < 0 || sy > h * 0.5) continue;
    const sr = 0.3 + rng() * 0.8;
    const b = 100 + Math.floor(rng() * 120);
    const tint = rng();
    const r = tint > 0.7 ? Math.min(255, b + 30) : b;
    const g = b;
    const bl = tint < 0.3 ? Math.min(255, b + 40) : b;
    ctx.fillStyle = `rgba(${r},${g},${bl},${0.3 + rng() * 0.5})`;
    ctx.beginPath();
    ctx.arc(sx, sy, sr, 0, Math.PI * 2);
    ctx.fill();
  }

  // Nebula patches — soft colored glows
  const nebulaColors = [
    { x: 0.15, y: 0.18, r: 60, g: 30, b: 80, size: 0.08 },
    { x: 0.72, y: 0.15, r: 40, g: 50, b: 90, size: 0.06 },
    { x: 0.88, y: 0.28, r: 70, g: 35, b: 55, size: 0.05 },
  ];
  for (const neb of nebulaColors) {
    const nx = neb.x * w, ny = neb.y * h, ns = neb.size * w;
    const nebGrad = ctx.createRadialGradient(nx, ny, 0, nx, ny, ns);
    nebGrad.addColorStop(0, `rgba(${neb.r},${neb.g},${neb.b},0.12)`);
    nebGrad.addColorStop(0.4, `rgba(${neb.r},${neb.g},${neb.b},0.06)`);
    nebGrad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = nebGrad;
    ctx.fillRect(nx - ns, ny - ns, ns * 2, ns * 2);
  }

  // Stars — large count with color variation
  for (let i = 0; i < 600; i++) {
    const sx = rng() * w;
    const sy = rng() * h * 0.52; // sky half only
    const sr = 0.3 + rng() * 1.8;
    const brightness = 140 + Math.floor(rng() * 115);
    const colorRoll = rng();
    let r = brightness, g = brightness, b = brightness;
    if (colorRoll < 0.15) { r = Math.min(255, brightness + 40); g = brightness - 20; b = brightness - 30; } // warm/red
    else if (colorRoll < 0.30) { b = Math.min(255, brightness + 50); r = brightness - 10; } // blue
    else if (colorRoll < 0.38) { r = Math.min(255, brightness + 20); g = Math.min(255, brightness + 10); } // yellow
    ctx.fillStyle = `rgba(${r},${g},${b},${0.4 + rng() * 0.6})`;
    ctx.beginPath();
    ctx.arc(sx, sy, sr, 0, Math.PI * 2);
    ctx.fill();
    // Star glow for brighter ones
    if (sr > 1.2) {
      const glowGrad = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr * 3);
      glowGrad.addColorStop(0, `rgba(${r},${g},${b},0.15)`);
      glowGrad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = glowGrad;
      ctx.beginPath();
      ctx.arc(sx, sy, sr * 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Moon — brighter and more prominent
  const moonX = size * 3;
  const moonY = size * 0.22;
  // Outer halo
  const moonHalo = ctx.createRadialGradient(moonX, moonY, 0, moonX, moonY, size * 0.45);
  moonHalo.addColorStop(0, 'rgba(180,195,230,0.25)');
  moonHalo.addColorStop(0.12, 'rgba(120,140,190,0.12)');
  moonHalo.addColorStop(0.35, 'rgba(60,80,130,0.05)');
  moonHalo.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = moonHalo;
  ctx.fillRect(0, 0, w, h);
  // Inner glow
  const moonGrad = ctx.createRadialGradient(moonX, moonY, 0, moonX, moonY, 20);
  moonGrad.addColorStop(0, 'rgba(250,248,240,0.85)');
  moonGrad.addColorStop(0.5, 'rgba(230,232,240,0.6)');
  moonGrad.addColorStop(1, 'rgba(180,190,210,0)');
  ctx.fillStyle = moonGrad;
  ctx.beginPath();
  ctx.arc(moonX, moonY, 20, 0, Math.PI * 2);
  ctx.fill();
  // Moon disc
  ctx.fillStyle = 'rgba(240,238,230,0.8)';
  ctx.beginPath();
  ctx.arc(moonX, moonY, 10, 0, Math.PI * 2);
  ctx.fill();

  // Horizon glow — warm-cool transition
  const horizonGrad = ctx.createLinearGradient(0, h * 0.42, 0, h * 0.58);
  horizonGrad.addColorStop(0, 'rgba(15,18,40,0)');
  horizonGrad.addColorStop(0.3, 'rgba(25,30,55,0.15)');
  horizonGrad.addColorStop(0.5, 'rgba(30,35,60,0.20)');
  horizonGrad.addColorStop(0.7, 'rgba(20,22,42,0.12)');
  horizonGrad.addColorStop(1, 'rgba(10,12,25,0)');
  ctx.fillStyle = horizonGrad;
  ctx.fillRect(0, 0, w, h);

  const tex = new THREE.CanvasTexture(canvas);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

let nightEnvMap = null;
let savedDayEnv = null;
let savedDayBg = null;

const HDR_ENVIRONMENTS = [
  { name: 'City 4', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/neuer_zollhof_1k.hdr' },
  { name: 'Venice', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/venice_sunset_1k.hdr' },
  { name: 'City 5', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/sunset_jhbcentral_1k.hdr' },
  { name: 'Venice Dawn', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/venice_dawn_2_1k.hdr' },
  { name: 'City 3', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/cedar_bridge_sunset_1_1k.hdr' },
  { name: 'City 2', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/evening_museum_courtyard_1k.hdr' },
  { name: 'Meadow', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/meadow_2_1k.hdr' },
  { name: 'Studio', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_09_1k.hdr' },
  { name: 'Forest', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/syferfontein_0d_clear_puresky_1k.hdr' },
  { name: 'Urban', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/potsdamer_platz_1k.hdr' },
  { name: 'Sunset', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/kloofendal_48d_partly_cloudy_puresky_1k.hdr' },
  { name: 'Warehouse', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/empty_warehouse_01_1k.hdr' },
  { name: 'Night', url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/moonlit_golf_1k.hdr' },
];

const hdrCache = {};
let currentHdrIndex = 1;

function loadHDR(index) {
  currentHdrIndex = index;
  const entry = HDR_ENVIRONMENTS[index];
  document.querySelectorAll('.hdr-btn').forEach((btn, i) => {
    btn.classList.toggle('active', i === index);
  });
  if (hdrCache[entry.url]) {
    scene.environment = hdrCache[entry.url];
    scene.background = hdrCache[entry.url];
    return;
  }
  rgbeLoader.load(entry.url, (hdr) => {
    const envMap = pmremGenerator.fromEquirectangular(hdr).texture;
    hdr.dispose();
    hdrCache[entry.url] = envMap;
    if (currentHdrIndex === index) {
      scene.environment = envMap;
      scene.background = envMap;
    }
  }, undefined, () => {});
}

// HDR loading is deferred — triggered after first render completes (see animation loop)

const normalMap = generateNormalMap();
const woodNormalMap = generateWoodGrainNormal(77);
const woodBumpMap = generateWoodBumpMap(133);
const clearcoatNormalMap = generateClearcoatNormalMap(180, 90, 4);

// ─── Wood Colors ───
const WOOD_COLORS = {
  birch: '#ffffff',
  pine: '#C4A46C',
  rail: '#a3a3a3',
  dark: '#6B5340',
};

// Single shared wood color map (neutral base) — tinted per-material via .color
const sharedWoodColorMap = generateWoodColorMap('#C8B090', 101);
const woodColorMaps = {
  birch: sharedWoodColorMap,
  pine: sharedWoodColorMap,
  rail: sharedWoodColorMap,
  dark: sharedWoodColorMap,
  peg: sharedWoodColorMap,
  red: sharedWoodColorMap,
  darkRed: sharedWoodColorMap,
  black: sharedWoodColorMap,
  charcoal: sharedWoodColorMap,
  tan: sharedWoodColorMap,
  darkWheel: sharedWoodColorMap,
  hole: sharedWoodColorMap,
};

// ─── Night mode state ───
let isNightMode = false;

// Store day lighting values for restoration
const dayLightSettings = {
  keyIntensity: 2.2,
  keyColor: 0xffeedd,
  fillIntensity: 0.75,
  fillSkyColor: 0xffecd2,
  fillGroundColor: 0x3a2f1a,
  rimIntensity: 2.2,
  rimColor: 0xffa94d,
  exposure: 0.50,
};

const nightLightSettings = {
  keyIntensity: 0.8,
  keyColor: 0x8899cc,
  fillIntensity: 0.7,
  fillSkyColor: 0x3a4a77,
  fillGroundColor: 0x1a1e2a,
  rimIntensity: 0.8,
  rimColor: 0x5577aa,
  exposure: 0.55,
};

// Train light references (populated in createTrain)
let trainHeadlight = null;
let trainHeadlightTarget = null;
let trainCabinLight = null;
let trainRearLight = null;
let trainFrontHubLight = null;
let trainFrontHubMat = null;
let trainHeadlightLensMat = null;
let trainCabinWindowMats = [];

// Train material references (populated in createTrain, used by UI bindings)
let trainMats = {
  body: null, cabin: null, roof: null, stack: null,
  stackCap: null, wheel: null, trim: null
};
function getAllTrainWoodMats() {
  return Object.values(trainMats).filter(Boolean);
}

// ─── Train state ───
let trainRunning = true;
let trainPaused = false; // true when user manually paused the train
let trainStopped = false; // true when train reached end of disconnected track
let trainGroup = null;
let trainPathPoints = [];
let trainT = 0;
const TRAIN_SPEED = 0.036;

// ─── Track Piece Definitions ───
// Each track piece defines: geometry creation, connectors (local positions + directions)

const GRID_SIZE = 4;           // grid cell size – all tracks fit on this grid
const TRACK_RAIL_H = 0.12;
const TRACK_RAIL_W = 0.12;
const TRACK_BED_H = 0.22;
const TRACK_GAUGE = 1.0; // distance between rails
const TIE_SPACING = 0.5;

function createWoodMaterial(color, roughness = 0.35, colorMapKey = null) {
  const c = new THREE.Color(color);
  const opts = {
    color: c,
    roughness,
    metalness: 0.0,
    envMapIntensity: 0.8,
    // PBR wood maps
    normalMap: woodNormalTex,
    normalScale: new THREE.Vector2(0.6, 0.6),
    bumpMap: woodBumpTex,
    bumpScale: 0.12,
    roughnessMap: woodRoughnessTex,
    // Clearcoat for lacquered wood look
    clearcoat: 0.45,
    clearcoatRoughness: 0.3,
    clearcoatNormalMap: clearcoatNormalTex,
    clearcoatNormalScale: new THREE.Vector2(0.15, 0.15),
    sheen: 0.3,
    sheenRoughness: 0.5,
    sheenColor: new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.3),
  };
  if (colorMapKey && woodColorMaps[colorMapKey]) {
    opts.map = woodColorMaps[colorMapKey];
  }
  return new THREE.MeshPhysicalMaterial(opts);
}

const birchMat = createWoodMaterial(WOOD_COLORS.birch, 0.32, 'birch');
const pineMat = createWoodMaterial(WOOD_COLORS.pine, 0.38, 'pine');
const railMat = createWoodMaterial(WOOD_COLORS.rail, 0.28, 'rail');
const darkMat = createWoodMaterial(WOOD_COLORS.dark, 0.4, 'dark');

// Connector peg material
const pegMat = createWoodMaterial('#A08060', 0.35, 'peg');

// Shared geometries (created once, reused everywhere via InstancedMesh)
// Convert to non-indexed to avoid WebGPU setIndexBuffer errors with shared buffers
const sharedTieGeo = new RoundedBoxGeometry(TRACK_GAUGE + 0.5, TRACK_BED_H, 0.3, 2, 0.04).toNonIndexed();
const sharedWideTieGeo = new RoundedBoxGeometry(TRACK_GAUGE + 0.8, TRACK_BED_H, 0.3, 2, 0.04).toNonIndexed();
const sharedPegGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.2, 8).toNonIndexed();
const sharedHoleGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.15, 8).toNonIndexed();
const sharedHoleMat = createWoodMaterial('#7A6545', 0.5, 'hole');

const _dummy = new THREE.Object3D();

// Helper: create a non-indexed RoundedBoxGeometry for WebGPU compatibility
function safeRoundedBox(w, h, d, seg, rad) {
  return new RoundedBoxGeometry(w, h, d, seg, rad).toNonIndexed();
}

function createTieInstances(geo, mat, positions, group, name) {
  const count = positions.length;
  // geo is already non-indexed; clone to give each InstancedMesh its own buffer
  const clonedGeo = geo.clone();
  const im = new THREE.InstancedMesh(clonedGeo, mat, count);
  im.castShadow = true;
  im.receiveShadow = true;
  im.name = name;
  for (let i = 0; i < count; i++) {
    const p = positions[i];
    _dummy.position.set(p.x, p.y, p.z);
    _dummy.rotation.set(p.rx || 0, p.ry || 0, p.rz || 0);
    _dummy.scale.set(1, 1, 1);
    _dummy.updateMatrix();
    im.setMatrixAt(i, _dummy.matrix);
  }
  im.instanceMatrix.needsUpdate = true;
  im.computeBoundingSphere();
  group.add(im);
  return im;
}

function createStraightTrack() {
  const group = new THREE.Group();
  const length = 3.5;

  // Ties as InstancedMesh
  const numTies = Math.floor(length / TIE_SPACING);
  const tiePositions = [];
  for (let i = 0; i <= numTies; i++) {
    tiePositions.push({ x: 0, y: TRACK_BED_H / 2, z: -length / 2 + i * TIE_SPACING });
  }
  createTieInstances(sharedTieGeo, birchMat, tiePositions, group, 'straight_ties');

  // Rails
  const railGeo = new RoundedBoxGeometry(TRACK_RAIL_W, TRACK_RAIL_H, length, 2, 0.02);
  const leftRail = new THREE.Mesh(railGeo, railMat);
  leftRail.position.set(-TRACK_GAUGE / 2, TRACK_BED_H + TRACK_RAIL_H / 2, 0);
  leftRail.castShadow = true;
  leftRail.name = 'straight_rail_left';
  group.add(leftRail);
  const rightRail = new THREE.Mesh(railGeo.clone(), railMat);
  rightRail.position.set(TRACK_GAUGE / 2, TRACK_BED_H + TRACK_RAIL_H / 2, 0);
  rightRail.castShadow = true;
  rightRail.name = 'straight_rail_right';
  group.add(rightRail);

  // Connector pegs at ends
  const peg1 = new THREE.Mesh(sharedPegGeo, pegMat);
  peg1.position.set(0, TRACK_BED_H + 0.1, -length / 2);
  peg1.name = 'straight_peg_front';
  group.add(peg1);

  const hole1 = new THREE.Mesh(sharedHoleGeo, sharedHoleMat);
  hole1.position.set(0, TRACK_BED_H + 0.08, length / 2);
  hole1.name = 'straight_hole_back';
  group.add(hole1);

  group.userData = {
    type: 'straight',
    length,
    connectors: [
      { pos: new THREE.Vector3(0, 0, -length / 2), dir: new THREE.Vector3(0, 0, -1) },
      { pos: new THREE.Vector3(0, 0, length / 2), dir: new THREE.Vector3(0, 0, 1) },
    ]
  };

  return group;
}

// Curve geometry helpers
// Arc starts at origin, initial tangent along +Z (matching straight exit at +Z facing +Z)
// For a LEFT turn: center is at (-radius, 0, 0), arc sweeps CCW in XZ
// For a RIGHT turn: center is at (+radius, 0, 0), arc sweeps CW in XZ
// sign: left = +1, right = -1

function curvePoint(angle, radius, sign) {
  // sign = +1 for left, -1 for right
  // Center at (sign * radius, 0, 0)
  // At angle=0, point = origin (0,0,0): center + radius * (-sign, 0, 0)
  // Arc parametrized so point rotates around center
  const cx = sign * radius;
  const x = cx + radius * (-sign * Math.cos(angle) + 0 * Math.sin(angle));
  const z = radius * Math.sin(angle);
  // At angle=0: x = cx + radius*(-sign) = sign*radius - sign*radius = 0 ✓
  // At angle=0: z = 0 ✓
  return new THREE.Vector3(x, 0, z);
}

function curveTangent(angle, radius, sign) {
  // Derivative of curvePoint w.r.t. angle
  const cx = sign * radius;
  const dx = radius * (sign * Math.sin(angle));
  const dz = radius * Math.cos(angle);
  return new THREE.Vector3(dx, 0, dz).normalize();
}

function createCurveTrack(curveAngle = Math.PI / 2, radius = 3.5, direction = 'left') {
  // radius MUST equal straight track length (3.5) for closed circuits
  const group = new THREE.Group();
  const segments = 12;

  // sign: +1 for left turn, -1 for right turn
  const sign = direction === 'right' ? -1 : 1;

  // Ties along curve as InstancedMesh
  const tiePositions = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const a = t * curveAngle;
    const p = curvePoint(a, radius, sign);
    const tang = curveTangent(a, radius, sign);
    tiePositions.push({ x: p.x, y: TRACK_BED_H / 2, z: p.z, ry: Math.atan2(tang.x, tang.z) });
  }
  createTieInstances(sharedTieGeo, birchMat, tiePositions, group, 'curve_ties');

  // Rails as segmented pieces — use InstancedMesh per segment pair
  // Each segment has a unique length so we need individual geometries, but we can batch the 2 sides
  for (let i = 0; i < segments; i++) {
    const t0 = i / segments;
    const t1 = (i + 1) / segments;
    const a0 = t0 * curveAngle;
    const a1 = t1 * curveAngle;

    // Compute both sides
    const railInstances = [];
    for (const side of [-1, 1]) {
      const r = radius + side * TRACK_GAUGE / 2;
      const cx = sign * radius;
      const x0 = cx + r * (-sign * Math.cos(a0));
      const z0 = r * Math.sin(a0);
      const x1 = cx + r * (-sign * Math.cos(a1));
      const z1 = r * Math.sin(a1);
      const dx = x1 - x0, dz = z1 - z0;
      railInstances.push({ x: (x0 + x1) / 2, z: (z0 + z1) / 2, ry: Math.atan2(dx, dz), segLen: Math.sqrt(dx * dx + dz * dz) });
    }

    // Use each rail's actual segment length to avoid gaps on outer rail
    for (let s = 0; s < 2; s++) {
      const ri = railInstances[s];
      const railGeo = new RoundedBoxGeometry(TRACK_RAIL_W, TRACK_RAIL_H, ri.segLen, 2, 0.02);
      const railMesh = new THREE.Mesh(railGeo, railMat);
      railMesh.position.set(ri.x, TRACK_BED_H + TRACK_RAIL_H / 2, ri.z);
      railMesh.rotation.set(0, ri.ry, 0);
      railMesh.castShadow = true;
      railMesh.name = `curve_rail_seg_${i}_${s}`;
      group.add(railMesh);
    }
  }

  // Connector positions & directions
  const startPos = curvePoint(0, radius, sign);
  const endPos = curvePoint(curveAngle, radius, sign);
  const startDir = curveTangent(0, radius, sign).clone().negate();
  const endDir = curveTangent(curveAngle, radius, sign).clone();

  // Pegs
  const peg1 = new THREE.Mesh(sharedPegGeo, pegMat);
  peg1.position.copy(startPos).setY(TRACK_BED_H + 0.1);
  peg1.name = 'curve_peg_start';
  group.add(peg1);

  const hole1 = new THREE.Mesh(sharedHoleGeo, sharedHoleMat);
  hole1.position.copy(endPos).setY(TRACK_BED_H + 0.08);
  hole1.name = 'curve_hole_end';
  group.add(hole1);

  group.userData = {
    type: direction === 'right' ? 'curveRight' : 'curveLeft',
    curveAngle,
    radius,
    direction,
    connectors: [
      { pos: startPos, dir: startDir },
      { pos: endPos, dir: endDir },
    ]
  };

  return group;
}

function createBridgeTrack() {
  const group = new THREE.Group();
  const length = 3.5;
  const height = 1.32;

  // Bridge supports (4 pillars)
  const pillarGeo = new RoundedBoxGeometry(0.3, height, 0.3, 2, 0.04);
  let pi = 0;
  for (const xOff of [-0.7, 0.7]) {
    for (const zOff of [-length / 2 + 0.3, length / 2 - 0.3]) {
      const pillar = new THREE.Mesh(pi === 0 ? pillarGeo : pillarGeo.clone(), darkMat);
      pillar.position.set(xOff, height / 2, zOff);
      pillar.castShadow = true;
      pillar.receiveShadow = true;
      pillar.name = `bridge_pillar_${pi}`;
      group.add(pillar);
      pi++;
    }
  }

  // Side beams (2)
  const sideGeo = new RoundedBoxGeometry(0.15, 0.15, length, 2, 0.03);
  const sideL = new THREE.Mesh(sideGeo, darkMat);
  sideL.position.set(-0.7, height + 0.08, 0);
  sideL.castShadow = true;
  sideL.name = 'bridge_side_left';
  group.add(sideL);
  const sideR = new THREE.Mesh(sideGeo.clone(), darkMat);
  sideR.position.set(0.7, height + 0.08, 0);
  sideR.castShadow = true;
  sideR.name = 'bridge_side_right';
  group.add(sideR);

  // Track bed ties on top
  const numTies = Math.floor(length / TIE_SPACING);
  const tiePositions = [];
  for (let i = 0; i <= numTies; i++) {
    tiePositions.push({ x: 0, y: height + TRACK_BED_H / 2, z: -length / 2 + i * TIE_SPACING });
  }
  createTieInstances(sharedTieGeo, birchMat, tiePositions, group, 'bridge_ties');

  // Rails
  const railGeo = new RoundedBoxGeometry(TRACK_RAIL_W, TRACK_RAIL_H, length, 2, 0.02);
  const railL = new THREE.Mesh(railGeo, railMat);
  railL.position.set(-TRACK_GAUGE / 2, height + TRACK_BED_H + TRACK_RAIL_H / 2, 0);
  railL.castShadow = true;
  railL.name = 'bridge_rail_left';
  group.add(railL);
  const railR = new THREE.Mesh(railGeo.clone(), railMat);
  railR.position.set(TRACK_GAUGE / 2, height + TRACK_BED_H + TRACK_RAIL_H / 2, 0);
  railR.castShadow = true;
  railR.name = 'bridge_rail_right';
  group.add(railR);

  // Pegs
  const peg1 = new THREE.Mesh(sharedPegGeo, pegMat);
  peg1.position.set(0, height + TRACK_BED_H + 0.1, -length / 2);
  peg1.name = 'bridge_peg_front';
  group.add(peg1);

  const hole1 = new THREE.Mesh(sharedHoleGeo, sharedHoleMat);
  hole1.position.set(0, height + TRACK_BED_H + 0.08, length / 2);
  hole1.name = 'bridge_hole_back';
  group.add(hole1);

  group.userData = {
    type: 'bridge',
    length,
    height,
    connectors: [
      { pos: new THREE.Vector3(0, height, -length / 2), dir: new THREE.Vector3(0, 0, -1) },
      { pos: new THREE.Vector3(0, height, length / 2), dir: new THREE.Vector3(0, 0, 1) },
    ]
  };

  return group;
}

function createBridgeCurveTrack(curveAngle = Math.PI / 2, radius = 3.5, direction = 'left') {
  const group = new THREE.Group();
  const segments = 12;
  const height = 1.32; // match bridge height

  const sign = direction === 'right' ? -1 : 1;

  // Pillar supports along the curve
  const numPillars = 6;
  for (let i = 0; i < numPillars; i++) {
    const t = (i + 0.5) / numPillars;
    const a = t * curveAngle;
    const p = curvePoint(a, radius, sign);
    const tang = curveTangent(a, radius, sign);
    const perpX = -tang.z;
    const perpZ = tang.x;

    const pillarGeo = new RoundedBoxGeometry(0.25, height, 0.25, 2, 0.04);
    for (const side of [-0.7, 0.7]) {
      const pillar = new THREE.Mesh(pillarGeo.clone(), darkMat);
      pillar.position.set(p.x + perpX * side, height / 2, p.z + perpZ * side);
      pillar.castShadow = true;
      pillar.receiveShadow = true;
      pillar.name = `bcurve_pillar_${i}_${side > 0 ? 'r' : 'l'}`;
      group.add(pillar);
    }
  }

  // Side beams along the curve (guardrails)
  for (let i = 0; i < segments; i++) {
    const t0 = i / segments;
    const t1 = (i + 1) / segments;
    const a0 = t0 * curveAngle;
    const a1 = t1 * curveAngle;

    const p0 = curvePoint(a0, radius, sign);
    const p1 = curvePoint(a1, radius, sign);
    const tang0 = curveTangent(a0, radius, sign);
    const tang1 = curveTangent(a1, radius, sign);

    for (const side of [-0.7, 0.7]) {
      const perp0x = -tang0.z, perp0z = tang0.x;
      const perp1x = -tang1.z, perp1z = tang1.x;
      const sx = p0.x + perp0x * side;
      const sz = p0.z + perp0z * side;
      const ex = p1.x + perp1x * side;
      const ez = p1.z + perp1z * side;
      const dx = ex - sx, dz = ez - sz;
      const segLen = Math.sqrt(dx * dx + dz * dz);
      const beamGeo = new RoundedBoxGeometry(0.12, 0.12, segLen, 2, 0.02);
      const beam = new THREE.Mesh(beamGeo, darkMat);
      beam.position.set((sx + ex) / 2, height + 0.08, (sz + ez) / 2);
      beam.rotation.y = Math.atan2(dx, dz);
      beam.castShadow = true;
      beam.name = `bcurve_beam_${i}_${side > 0 ? 'r' : 'l'}`;
      group.add(beam);
    }
  }

  // Cross braces between pillars
  for (let i = 0; i < numPillars; i++) {
    const t = (i + 0.5) / numPillars;
    const a = t * curveAngle;
    const p = curvePoint(a, radius, sign);
    const tang = curveTangent(a, radius, sign);
    const perpX = -tang.z;
    const perpZ = tang.x;

    const braceWidth = 1.4;
    const braceGeo = new RoundedBoxGeometry(0.08, 0.08, braceWidth * 0.9, 2, 0.02);
    const brace = new THREE.Mesh(braceGeo, darkMat);
    brace.position.set(p.x, height * 0.55, p.z);
    brace.rotation.y = Math.atan2(perpX, perpZ);
    brace.castShadow = true;
    brace.name = `bcurve_brace_${i}`;
    group.add(brace);
  }

  // Ties along curve at bridge height
  const tiePositions = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const a = t * curveAngle;
    const p = curvePoint(a, radius, sign);
    const tang = curveTangent(a, radius, sign);
    tiePositions.push({ x: p.x, y: height + TRACK_BED_H / 2, z: p.z, ry: Math.atan2(tang.x, tang.z) });
  }
  createTieInstances(sharedTieGeo, birchMat, tiePositions, group, 'bcurve_ties');

  // Rails as segmented pieces at bridge height
  for (let i = 0; i < segments; i++) {
    const t0 = i / segments;
    const t1 = (i + 1) / segments;
    const a0 = t0 * curveAngle;
    const a1 = t1 * curveAngle;

    const railInstances = [];
    for (const side of [-1, 1]) {
      const r = radius + side * TRACK_GAUGE / 2;
      const cx = sign * radius;
      const x0 = cx + r * (-sign * Math.cos(a0));
      const z0 = r * Math.sin(a0);
      const x1 = cx + r * (-sign * Math.cos(a1));
      const z1 = r * Math.sin(a1);
      const dx = x1 - x0, dz = z1 - z0;
      railInstances.push({ x: (x0 + x1) / 2, z: (z0 + z1) / 2, ry: Math.atan2(dx, dz), segLen: Math.sqrt(dx * dx + dz * dz) });
    }

    for (let s = 0; s < 2; s++) {
      const ri = railInstances[s];
      const railGeo = new RoundedBoxGeometry(TRACK_RAIL_W, TRACK_RAIL_H, ri.segLen, 2, 0.02);
      const railMesh = new THREE.Mesh(railGeo, railMat);
      railMesh.position.set(ri.x, height + TRACK_BED_H + TRACK_RAIL_H / 2, ri.z);
      railMesh.rotation.set(0, ri.ry, 0);
      railMesh.castShadow = true;
      railMesh.name = `bcurve_rail_seg_${i}_${s}`;
      group.add(railMesh);
    }
  }

  // Connector positions & directions at bridge height
  const startPos = curvePoint(0, radius, sign);
  startPos.y = height;
  const endPos = curvePoint(curveAngle, radius, sign);
  endPos.y = height;
  const startDir = curveTangent(0, radius, sign).clone().negate();
  const endDir = curveTangent(curveAngle, radius, sign).clone();

  // Pegs
  const peg1 = new THREE.Mesh(sharedPegGeo, pegMat);
  peg1.position.copy(startPos).setY(height + TRACK_BED_H + 0.1);
  peg1.name = 'bcurve_peg_start';
  group.add(peg1);

  const hole1 = new THREE.Mesh(sharedHoleGeo, sharedHoleMat);
  hole1.position.copy(endPos).setY(height + TRACK_BED_H + 0.08);
  hole1.name = 'bcurve_hole_end';
  group.add(hole1);

  group.userData = {
    type: direction === 'right' ? 'bridgeCurveRight' : 'bridgeCurveLeft',
    curveAngle,
    radius,
    direction,
    height,
    connectors: [
      { pos: startPos, dir: startDir },
      { pos: endPos, dir: endDir },
    ]
  };

  return group;
}

function createSlopeTrack() {
  const group = new THREE.Group();
  const length = 3.5;
  const height = 1.32; // match bridge height
  const slopeAngle = Math.atan2(height, length); // incline angle

  // Support structure — graduated pillars underneath the ramp
  const numSupports = 5;
  for (let i = 0; i < numSupports; i++) {
    const t = (i + 1) / (numSupports + 1);
    const z = -length / 2 + t * length;
    const h = t * height;
    if (h < 0.15) continue;
    const pillarGeo = new RoundedBoxGeometry(0.25, h, 0.25, 2, 0.04);
    for (const xOff of [-0.55, 0.55]) {
      const pillar = new THREE.Mesh(pillarGeo, darkMat);
      pillar.position.set(xOff, h / 2, z);
      pillar.castShadow = true;
      pillar.receiveShadow = true;
      pillar.name = `slope_pillar_${i}_${xOff > 0 ? 'r' : 'l'}`;
      group.add(pillar);
    }
  }

  // Cross braces between pillars for structural look
  for (let i = 0; i < numSupports; i++) {
    const t = (i + 1) / (numSupports + 1);
    const z = -length / 2 + t * length;
    const h = t * height;
    if (h < 0.3) continue;
    const braceGeo = new RoundedBoxGeometry(0.08, 0.08, 1.1, 2, 0.02);
    const brace = new THREE.Mesh(braceGeo, darkMat);
    brace.position.set(0, h * 0.45, z);
    brace.castShadow = true;
    brace.name = `slope_brace_${i}`;
    group.add(brace);
  }

  // Side rails (angled guard rails on each side of the ramp)
  const sideLen = Math.sqrt(length * length + height * height);
  const sideGeo = new RoundedBoxGeometry(0.08, 0.15, sideLen, 2, 0.02);
  for (const xOff of [-0.65, 0.65]) {
    const side = new THREE.Mesh(sideGeo, darkMat);
    side.position.set(xOff, height / 2 + 0.08, 0);
    side.rotation.x = -slopeAngle;
    side.castShadow = true;
    side.name = `slope_side_${xOff > 0 ? 'right' : 'left'}`;
    group.add(side);
  }

  // Ties along the slope (tilted to follow the incline)
  const numTies = Math.floor(length / TIE_SPACING);
  const tiePositions = [];
  for (let i = 0; i <= numTies; i++) {
    const t = i / numTies;
    const z = -length / 2 + t * length;
    const y = t * height + TRACK_BED_H / 2;
    tiePositions.push({ x: 0, y, z, rx: -slopeAngle });
  }
  createTieInstances(sharedTieGeo, birchMat, tiePositions, group, 'slope_ties');

  // Rails along the slope — segmented to follow incline
  const railSegments = 8;
  for (let i = 0; i < railSegments; i++) {
    const t0 = i / railSegments;
    const t1 = (i + 1) / railSegments;
    const z0 = -length / 2 + t0 * length;
    const z1 = -length / 2 + t1 * length;
    const y0 = t0 * height + TRACK_BED_H + TRACK_RAIL_H / 2;
    const y1 = t1 * height + TRACK_BED_H + TRACK_RAIL_H / 2;
    const dz = z1 - z0;
    const dy = y1 - y0;
    const segLen = Math.sqrt(dz * dz + dy * dy);

    for (const side of [-1, 1]) {
      const railGeo = new RoundedBoxGeometry(TRACK_RAIL_W, TRACK_RAIL_H, segLen, 2, 0.02);
      const rail = new THREE.Mesh(railGeo, railMat);
      rail.position.set(side * TRACK_GAUGE / 2, (y0 + y1) / 2, (z0 + z1) / 2);
      rail.rotation.x = -slopeAngle;
      rail.castShadow = true;
      rail.name = `slope_rail_${i}_${side > 0 ? 'r' : 'l'}`;
      group.add(rail);
    }
  }

  // Pegs & holes — connector 0 at ground (front), connector 1 at bridge height (back)
  const peg1 = new THREE.Mesh(sharedPegGeo, pegMat);
  peg1.position.set(0, TRACK_BED_H + 0.1, -length / 2);
  peg1.name = 'slope_peg_front';
  group.add(peg1);

  const hole1 = new THREE.Mesh(sharedHoleGeo, sharedHoleMat);
  hole1.position.set(0, height + TRACK_BED_H + 0.08, length / 2);
  hole1.name = 'slope_hole_back';
  group.add(hole1);

  group.userData = {
    type: 'slope',
    length,
    height,
    connectors: [
      { pos: new THREE.Vector3(0, 0, -length / 2), dir: new THREE.Vector3(0, 0, -1) },          // ground level
      { pos: new THREE.Vector3(0, height, length / 2), dir: new THREE.Vector3(0, 0, 1) },  // bridge level
    ]
  };

  return group;
}

function createCrossing() {
  const group = new THREE.Group();
  const length = 3.5; // Match straight track size

  // Ties in Z direction
  const numTiesZ = Math.floor(length / TIE_SPACING);
  const tiePositionsZ = [];
  for (let i = 0; i <= numTiesZ; i++) {
    tiePositionsZ.push({ x: 0, y: TRACK_BED_H / 2, z: -length / 2 + i * TIE_SPACING });
  }
  createTieInstances(sharedTieGeo, birchMat, tiePositionsZ, group, 'crossing_ties_z');

  // Ties in X direction
  const numTiesX = Math.floor(length / TIE_SPACING);
  const tiePositionsX = [];
  for (let i = 0; i <= numTiesX; i++) {
    const xPos = -length / 2 + i * TIE_SPACING;
    // Skip ties that overlap with the center area where Z-ties already exist
    if (Math.abs(xPos) < TRACK_GAUGE / 2 + 0.3) continue;
    tiePositionsX.push({ x: xPos, y: TRACK_BED_H / 2, z: 0, ry: Math.PI / 2 });
  }
  createTieInstances(sharedTieGeo, birchMat, tiePositionsX, group, 'crossing_ties_x');

  // Rails in both directions — 4 rails
  const railGeo = new RoundedBoxGeometry(TRACK_RAIL_W, TRACK_RAIL_H, length, 2, 0.02);
  let ri = 0;
  for (const axis of ['x', 'z']) {
    for (const off of [-TRACK_GAUGE / 2, TRACK_GAUGE / 2]) {
      const rail = new THREE.Mesh(ri === 0 ? railGeo : railGeo.clone(), railMat);
      if (axis === 'z') {
        rail.position.set(off, TRACK_BED_H + TRACK_RAIL_H / 2, 0);
      } else {
        rail.position.set(0, TRACK_BED_H + TRACK_RAIL_H / 2, off);
        rail.rotation.set(0, Math.PI / 2, 0);
      }
      rail.castShadow = true;
      rail.name = `crossing_rail_${ri}`;
      group.add(rail);
      ri++;
    }
  }

  // Connector pegs/holes at ends
  const pegZ0 = new THREE.Mesh(sharedPegGeo, pegMat);
  pegZ0.position.set(0, TRACK_BED_H + 0.1, -length / 2);
  pegZ0.name = 'crossing_peg_z0';
  group.add(pegZ0);
  const holeZ1 = new THREE.Mesh(sharedHoleGeo, sharedHoleMat);
  holeZ1.position.set(0, TRACK_BED_H + 0.08, length / 2);
  holeZ1.name = 'crossing_hole_z1';
  group.add(holeZ1);
  const pegX0 = new THREE.Mesh(sharedPegGeo, pegMat);
  pegX0.position.set(-length / 2, TRACK_BED_H + 0.1, 0);
  pegX0.name = 'crossing_peg_x0';
  group.add(pegX0);
  const holeX1 = new THREE.Mesh(sharedHoleGeo, sharedHoleMat);
  holeX1.position.set(length / 2, TRACK_BED_H + 0.08, 0);
  holeX1.name = 'crossing_hole_x1';
  group.add(holeX1);

  group.userData = {
    type: 'crossing',
    connectors: [
      { pos: new THREE.Vector3(0, 0, -length / 2), dir: new THREE.Vector3(0, 0, -1) },
      { pos: new THREE.Vector3(0, 0, length / 2), dir: new THREE.Vector3(0, 0, 1) },
      { pos: new THREE.Vector3(-length / 2, 0, 0), dir: new THREE.Vector3(-1, 0, 0) },
      { pos: new THREE.Vector3(length / 2, 0, 0), dir: new THREE.Vector3(1, 0, 0) },
    ]
  };

  return group;
}

// Simple concrete — no heavy procedural generation
const concreteNormalMap = generateNormalMap();
const concreteRoughnessMap = (() => {
  const { canvas, ctx } = createCanvas(4, 4);
  ctx.fillStyle = 'rgb(180,180,180)';
  ctx.fillRect(0, 0, 4, 4);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
})();
const concreteColorMap = (() => {
  const { canvas, ctx } = createCanvas(4, 4);
  ctx.fillStyle = '#8A8680';
  ctx.fillRect(0, 0, 4, 4);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
})();

// ─── Ground Plane ───
const groundGeo = new RoundedBoxGeometry(24, 0.2, 24, 2, 0.03);
groundGeo.translate(0, -0.1, 0);

// PBR Concrete textures from Poly Haven (concrete_floor_worn_001)
const texLoader = new THREE.TextureLoader();
const PH = 'https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/concrete_floor_worn_001/';

function loadTex(map, srgb) {
  const t = texLoader.load(PH + 'concrete_floor_worn_001_' + map + '_1k.jpg');
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(4, 4);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const concreteColor = loadTex('diff', true);
const concreteNormal = loadTex('nor_gl', false);
const concreteRoughness = loadTex('rough', false);
const concreteAO = loadTex('ao', false);
const concreteDisp = loadTex('disp', false);
const concreteArm = loadTex('arm', false);

const groundMat = new THREE.MeshPhysicalMaterial({
  map: concreteColor,
  normalMap: concreteNormal,
  normalScale: new THREE.Vector2(1.37, 1.37),
  roughnessMap: concreteRoughness,
  roughness: 0.22,
  metalnessMap: concreteArm,
  metalness: 0.0,
  aoMap: concreteAO,
  aoMapIntensity: 1.27,
  displacementMap: concreteDisp,
  displacementScale: 0.02,
  envMapIntensity: 1.2,
  // Wet concrete look
  clearcoat: 0.6,
  clearcoatRoughness: 0.53,
  clearcoatNormalMap: concreteNormal,
  clearcoatNormalScale: new THREE.Vector2(0.3, 0.3),
  sheen: 0.22,
  color: '#9a9590',
  reflectivity: 0.61,
  ior: 1.39,
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.receiveShadow = true;
ground.name = 'ground';
scene.add(ground);

// ─── Lighting ───
const keyLight = new THREE.DirectionalLight(0xffeedd, 2.2);
keyLight.position.set(6, 10, 4);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(512, 512);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 30;
keyLight.shadow.camera.left = -14;
keyLight.shadow.camera.right = 14;
keyLight.shadow.camera.top = 14;
keyLight.shadow.camera.bottom = -14;
keyLight.shadow.bias = -0.0005;
keyLight.shadow.normalBias = 0.04;
keyLight.shadow.radius = 2;
keyLight.shadow.blurSamples = 4;
keyLight.name = 'keyLight';
scene.add(keyLight);

const fillLight = new THREE.HemisphereLight(0xffecd2, 0x3a2f1a, 0.75);
fillLight.name = 'fillLight';
scene.add(fillLight);

const rimLight = new THREE.SpotLight(0xffa94d, 2.2, 25, Math.PI / 6, 0.5, 1);
rimLight.position.set(-4, 6, -5);
rimLight.target.position.set(0, 0, 0);
rimLight.name = 'rimLight';
scene.add(rimLight);
scene.add(rimLight.target);

// Night ambient light — provides base illumination so night isn't pitch black
const nightAmbient = new THREE.AmbientLight(0x3344660, 0);
nightAmbient.name = 'nightAmbient';
scene.add(nightAmbient);

// ─── SSR Uniforms ───
const ssrEnabledU    = uniform(1.0);  // Enabled by default
const ssrStrengthU   = uniform(1.00);
const ssrThicknessU  = uniform(0.21);
const ssrMaxDistU    = uniform(1.60);
const ssrFresnelPowU = uniform(1.57);
const ssrFadeU       = uniform(0.90);
const projMatU       = uniform(camera.projectionMatrix);
const projInvMatU    = uniform(camera.projectionMatrixInverse);
let ssrMarchSteps    = 16;  // Configurable — triggers pipeline rebuild
let ssrRefineSteps   = 1;   // Configurable — triggers pipeline rebuild
let renderScale      = 1.0; // Resolution scale multiplier

// ─── Post-processing ───
const postProcessing = new THREE.PostProcessing(renderer);
const grainEnabledU = uniform(0.0);
const grainIntensityU = uniform(0.07);
const highlightClampU = uniform(0.0);
const exposureU = uniform(0.50);
const contrastU = uniform(0.0);
let currentToneMapping = THREE.ACESFilmicToneMapping;
let toneMappingEnabled = true;
const scenePass = pass(scene, camera);

scenePass.setMRT(mrt({
  output: output,
  normal: transformedNormalView,
}));

// ─── AO ───
const aoEnabledU = uniform(1.0);

const scenePassColor = scenePass.getTextureNode('output');
const scenePassNormal = scenePass.getTextureNode('normal');
const scenePassDepth = scenePass.getTextureNode('depth');

const aoPass = ao(scenePassDepth, scenePassNormal, camera);
aoPass.distanceExponent.value = 1.03;
aoPass.distanceFallOff.value = 0.25;
aoPass.radius.value = 0.10;
aoPass.scale.value = 0.60;
aoPass.thickness.value = 0.71;
aoPass.samples.value = 4;

const aoTexture = aoPass.getTextureNode();
const aoFactor = vec3(aoTexture.x, aoTexture.x, aoTexture.x);

const aoMixed = aoFactor.mul(aoEnabledU).add(float(1.0).sub(aoEnabledU));
const aoBlended = scenePassColor.mul(aoMixed);

// ─── SSR Node (rebuilt with configurable step counts) ───
function buildSsrNode(marchSteps, refineSteps) {
  return Fn(([colorTex, depthTex, normalTex]) => {
    const uv = screenUV;
    const rawDepth  = depthTex.sample(uv).x;
    const isSky     = rawDepth.greaterThanEqual(0.999);

    const A    = projMatU.element(2).element(2);
    const B    = projMatU.element(3).element(2);
    const ndcZ = rawDepth.mul(2.0).sub(1.0);
    const linZ = B.div(ndcZ.add(A));

    const clipX = uv.x.mul(2.0).sub(1.0);
    const clipY = float(1.0).sub(uv.y).mul(2.0).sub(1.0);
    const vx = clipX.mul(projInvMatU.element(0).element(0)).mul(linZ);
    const vy = clipY.mul(projInvMatU.element(1).element(1)).mul(linZ);
    const viewPos = vec3(vx, vy, linZ.negate());

    const normN      = normalTex.sample(uv).xyz.normalize();
    const viewDir    = viewPos.normalize();
    const reflDirNorm = viewDir.sub(normN.mul(viewDir.dot(normN).mul(2.0))).normalize();

    const NdotV   = normN.dot(viewDir.negate()).clamp(0.0, 1.0);
    const fresnel = float(1.0).sub(NdotV).pow(ssrFresnelPowU).clamp(0.0, 1.0);
    const reflZ   = reflDirNorm.z;

    const hitColor  = vec3(0.0, 0.0, 0.0).toVar();
    const hitWeight = float(0.0).toVar();
    const hitT      = float(0.0).toVar();
    const prevT     = float(0.0).toVar();

    If(ssrEnabledU.greaterThan(0.5).and(reflZ.lessThan(0.1)).and(isSky.not()), () => {
      Loop(marchSteps, ({ i }) => {
        const fi = float(i).add(1.0);
        const t  = fi.div(float(marchSteps)).mul(ssrMaxDistU);
        const samplePos = viewPos.add(reflDirNorm.mul(t));

        const negZ   = samplePos.z.negate();
        const sClipX = samplePos.x.mul(projMatU.element(0).element(0)).div(negZ);
        const sClipY = samplePos.y.mul(projMatU.element(1).element(1)).div(negZ);
        const sUV = vec2(
          sClipX.mul(0.5).add(0.5),
          float(1.0).sub(sClipY.mul(0.5).add(0.5))
        );

        const inBounds = sUV.x.greaterThanEqual(0.0).and(sUV.x.lessThanEqual(1.0))
          .and(sUV.y.greaterThanEqual(0.0)).and(sUV.y.lessThanEqual(1.0));

        If(inBounds.and(hitWeight.lessThan(0.5)), () => {
          const sampledDepth = depthTex.sample(sUV).x;
          const sampledNdcZ  = sampledDepth.mul(2.0).sub(1.0);
          const sampledLinZ  = B.div(sampledNdcZ.add(A));
          const diff   = negZ.sub(sampledLinZ);
          const isHit  = diff.greaterThan(0.0).and(diff.lessThan(ssrThicknessU));
          const notSky = sampledDepth.lessThan(0.999);

          If(isHit.and(notSky), () => {
            hitT.assign(t);
            hitWeight.assign(1.0);
          });
        });

        If(hitWeight.lessThan(0.5), () => {
          prevT.assign(t);
        });
      });

      If(hitWeight.greaterThan(0.5), () => {
        const loT = prevT.toVar();
        const hiT = hitT.toVar();

        Loop(refineSteps, () => {
          const midT   = loT.add(hiT).mul(0.5);
          const midPos = viewPos.add(reflDirNorm.mul(midT));

          const midNegZ  = midPos.z.negate();
          const midClipX = midPos.x.mul(projMatU.element(0).element(0)).div(midNegZ);
          const midClipY = midPos.y.mul(projMatU.element(1).element(1)).div(midNegZ);
          const midUV = vec2(
            midClipX.mul(0.5).add(0.5),
            float(1.0).sub(midClipY.mul(0.5).add(0.5))
          );

          const midDepth = depthTex.sample(midUV).x;
          const midNdcZ  = midDepth.mul(2.0).sub(1.0);
          const midLinZ  = B.div(midNdcZ.add(A));
          const midDiff  = midNegZ.sub(midLinZ);

          If(midDiff.greaterThan(0.0), () => {
            hiT.assign(midT);
          }).Else(() => {
            loT.assign(midT);
          });
        });

        const finalT   = loT.add(hiT).mul(0.5);
        const finalPos = viewPos.add(reflDirNorm.mul(finalT));

        const finalNegZ  = finalPos.z.negate();
        const finalClipX = finalPos.x.mul(projMatU.element(0).element(0)).div(finalNegZ);
        const finalClipY = finalPos.y.mul(projMatU.element(1).element(1)).div(finalNegZ);
        const finalUV = vec2(
          finalClipX.mul(0.5).add(0.5),
          float(1.0).sub(finalClipY.mul(0.5).add(0.5))
        );

        const edgeX    = finalUV.x.mul(float(1.0).sub(finalUV.x)).mul(4.0).clamp(0.0, 1.0);
        const edgeY    = finalUV.y.mul(float(1.0).sub(finalUV.y)).mul(4.0).clamp(0.0, 1.0);
        const edgeFade = edgeX.mul(edgeY);
        const distFade = float(1.0).sub(finalT.div(ssrMaxDistU)).clamp(0.0, 1.0);

        const sampledColor = colorTex.sample(finalUV).xyz;
        hitColor.assign(sampledColor.mul(edgeFade).mul(distFade));
      });
    });

    const reflectionMix = hitWeight.mul(fresnel).mul(ssrStrengthU).mul(ssrFadeU);
    return vec4(hitColor, reflectionMix);
  });
}

// ─── Bloom ───
let bloomEnabled = false;
const bloomStrengthU = uniform(0.35);
const bloomRadiusU = uniform(0.4);
const bloomThresholdU = uniform(0.85);

// ─── Depth of Field ───
let dofEnabled = false;
const focusDistU = uniform(8.3);
const focalLengthU = uniform(5.0);
const bokehScaleU = uniform(10.0);

let _rebuildQueued = false;
function rebuildPostPipeline() {
  // Debounce rapid pipeline rebuilds — WebGPU shader compilation is async and overlapping
  // rebuilds can cause GPU stalls / freezes
  if (_rebuildQueued) return;
  _rebuildQueued = true;
  queueMicrotask(() => {
    _rebuildQueued = false;
    try {
      _doRebuildPostPipeline();
    } catch (e) {
      console.warn('Pipeline rebuild error:', e);
    }
  });
}

function _doRebuildPostPipeline() {
  const ssrReflectionNode = buildSsrNode(ssrMarchSteps, ssrRefineSteps);
  const ssrResult = ssrReflectionNode(scenePassColor, scenePassDepth, scenePassNormal);
  let source = aoBlended.add(vec4(ssrResult.xyz.mul(ssrResult.w), 0.0));

  if (bloomEnabled) {
    const bloomOnly = bloom(source, bloomStrengthU, bloomRadiusU, bloomThresholdU);
    source = source.add(bloomOnly);
  }

  if (dofEnabled) {
    const sceneViewZ = scenePass.getViewZNode();
    source = dof(source, sceneViewZ, focusDistU, focalLengthU, bokehScaleU);
  }

  const _uv = screenUV.mul(vec2(float(window.innerWidth), float(window.innerHeight)));
  const _seed = _uv.add(vec2(time.mul(7.23), time.mul(3.91)));
  const _noise = hash(_seed.dot(vec2(127.1, 311.7)));
  const _grainAmount = _noise.sub(0.5).mul(grainIntensityU).mul(grainEnabledU);

  const _shoulderThreshold = float(0.68).sub(highlightClampU.mul(0.25));
  const _compressed = source.sub(_shoulderThreshold).div(float(1.0).sub(_shoulderThreshold)).mul(float(1.0).sub(_shoulderThreshold).mul(float(1.0).sub(highlightClampU.mul(0.55)))).add(_shoulderThreshold);
  const _shouldApply = source.greaterThan(_shoulderThreshold);
  const _clamped = vec3(
    _shouldApply.select(_compressed.x, source.x),
    _shouldApply.select(_compressed.y, source.y),
    _shouldApply.select(_compressed.z, source.z),
  );
  const _highlightMixed = highlightClampU.greaterThan(0.001).select(_clamped, source);
  const _withGrain = clamp(_highlightMixed.add(_grainAmount), 0.0, 1.0);

  // Apply contrast adjustment (shift midpoint 0.5, scale, shift back)
  const _contrastFactor = contrastU.mul(2.0).add(1.0); // maps 0→1, 0.5→2, 1→3
  const _contrasted = clamp(_withGrain.sub(0.5).mul(_contrastFactor).add(0.5), 0.0, 1.0);
  const _withContrast = mix(_withGrain, _contrasted, contrastU.greaterThan(0.001).select(float(1.0), float(0.0)));

  // Apply tone mapping + color space via renderOutput (renderer.toneMapping is bypassed by PostProcessing)
  const _toneMapped = toneMappingEnabled
    ? renderOutput(_withContrast, currentToneMapping, THREE.SRGBColorSpace)
    : renderOutput(_withContrast, THREE.NoToneMapping, THREE.SRGBColorSpace);

  postProcessing.outputNode = _toneMapped;
  postProcessing.needsUpdate = true;
}

// Pipeline state — build lazily after first few frames render
let pipelineReady = false;
let frameCount = 0;
const PIPELINE_BUILD_FRAME = 3; // build after this many frames rendered

// ─── Track Assembly State ───
const placedTracks = [];
let selectedTrackType = null;
let ghostTrack = null;
let ghostRotation = 0; // accumulated rotation for ghost preview
let rotateTool = false; // rotate tool active state
let selectionTool = true; // selection tool active state (default)
let isDragging = false;
let dragTrack = null;
let dragOffset = new THREE.Vector3();
let hoveredTrack = null; // track under mouse (for highlight/delete)
let selectedPlacedTrack = null; // currently selected placed track (shows rotate button)
const rotatableTypes = new Set(['straight', 'curveLeft', 'curveRight', 'bridge', 'slope', 'bridgeCurveLeft', 'bridgeCurveRight']);
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

// Snap settings
const SNAP_DIST = 1.2;
const SNAP_ANGLE = Math.PI / 12;

// Base plate bounds – ground is 24×24 centered at origin
const PLATE_HALF = 12;
function isOnBasePlate(pos) {
  return Math.abs(pos.x) <= PLATE_HALF && Math.abs(pos.z) <= PLATE_HALF;
}
function clampToBasePlate(pos) {
  pos.x = Math.max(-PLATE_HALF, Math.min(PLATE_HALF, pos.x));
  pos.z = Math.max(-PLATE_HALF, Math.min(PLATE_HALF, pos.z));
  return pos;
}

// Highlight material for hovered track (for delete hint)
const highlightColor = new THREE.Color(0xff4444);
const originalMaterials = new WeakMap();

function createTrackByType(type) {
  switch (type) {
    case 'straight': return createStraightTrack();
    case 'curveLeft': return createCurveTrack(Math.PI / 2, 3.5, 'left');
    case 'curveRight': return createCurveTrack(Math.PI / 2, 3.5, 'right');
    case 'bridge': return createBridgeTrack();
    case 'slope': return createSlopeTrack();

    case 'crossing': return createCrossing();
    case 'bridgeCurveLeft': return createBridgeCurveTrack(Math.PI / 2, 3.5, 'left');
    case 'bridgeCurveRight': return createBridgeCurveTrack(Math.PI / 2, 3.5, 'right');
    default: return createStraightTrack();
  }
}

function getWorldConnectors(trackGroup) {
  const connectors = trackGroup.userData.connectors || [];
  return connectors.map(c => {
    const worldPos = c.pos.clone().applyMatrix4(trackGroup.matrixWorld);
    const worldDir = c.dir.clone().transformDirection(trackGroup.matrixWorld).normalize();
    return { pos: worldPos, dir: worldDir };
  });
}

function findSnapTarget(trackGroup) {
  const myConnectors = getWorldConnectors(trackGroup);
  let bestSnap = null;
  let bestDist = SNAP_DIST;

  for (let p = 0; p < placedTracks.length; p++) {
    const placed = placedTracks[p];
    if (placed === trackGroup) continue;
    const theirConnectors = getWorldConnectors(placed);
    for (let mi = 0; mi < myConnectors.length; mi++) {
      const mc = myConnectors[mi];
      for (let ti = 0; ti < theirConnectors.length; ti++) {
        const tc = theirConnectors[ti];
        const dist = mc.pos.distanceTo(tc.pos);
        if (dist < bestDist) {
          // Check that connectors face each other (opposite directions)
          const dot = mc.dir.dot(tc.dir);
          if (dot < -0.5) {
            bestDist = dist;
            bestSnap = { myConn: mc, theirConn: tc, myLocal: trackGroup.userData.connectors[mi], dist };
          }
        }
      }
    }
  }
  return bestSnap;
}

const _yAxis = new THREE.Vector3(0, 1, 0); // shared constant
const _snapOffset = new THREE.Vector3();
const _snapPivot = new THREE.Vector3();

// Auto-snap: finds best connector pair regardless of current rotation, returns the rotation angle needed
const AUTOSNAP_DIST = 1.6; // larger search radius for auto-snapping during drag/placement
function findAutoSnapTarget(trackGroup) {
  const myConnectors = getWorldConnectors(trackGroup);
  let bestSnap = null;
  let bestDist = AUTOSNAP_DIST;

  for (let p = 0; p < placedTracks.length; p++) {
    const placed = placedTracks[p];
    if (placed === trackGroup) continue;
    const theirConnectors = getWorldConnectors(placed);
    for (let mi = 0; mi < myConnectors.length; mi++) {
      const mc = myConnectors[mi];
      for (let ti = 0; ti < theirConnectors.length; ti++) {
        const tc = theirConnectors[ti];
        const dist = mc.pos.distanceTo(tc.pos);
        if (dist < bestDist) {
          bestDist = dist;
          bestSnap = { myConn: mc, theirConn: tc, myIdx: mi, dist };
        }
      }
    }
  }
  return bestSnap;
}

// Apply auto-snap: moves and rotates track to align connectors
function autoSnapTrack(trackGroup) {
  const snap = findAutoSnapTarget(trackGroup);
  if (!snap) return false;

  const theirWorldConn = snap.theirConn;
  const myLocalConn = trackGroup.userData.connectors[snap.myIdx];

  // Compute the rotation needed so that myConn.dir faces opposite to theirConn.dir
  // First get the local direction in world space accounting for current rotation
  const myWorldDir = myLocalConn.dir.clone().transformDirection(trackGroup.matrixWorld).normalize();
  const targetDirX = -theirWorldConn.dir.x;
  const targetDirZ = -theirWorldConn.dir.z;
  const currentAngle = Math.atan2(myWorldDir.x, myWorldDir.z);
  const targetAngle = Math.atan2(targetDirX, targetDirZ);
  let angleDiff = targetAngle - currentAngle;

  // Normalize to [-PI, PI]
  while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
  while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;

  // Snap the rotation to nearest 90° step
  const snappedAngleDiff = Math.round(angleDiff / (Math.PI / 2)) * (Math.PI / 2);

  // Apply rotation around the track's current center first
  const myWorldPos = myLocalConn.pos.clone().applyMatrix4(trackGroup.matrixWorld);
  trackGroup.rotation.y += snappedAngleDiff;
  trackGroup.updateMatrixWorld(true);

  // Now recompute my connector position after rotation and translate to align
  const newMyWorldPos = myLocalConn.pos.clone().applyMatrix4(trackGroup.matrixWorld);
  _snapOffset.copy(theirWorldConn.pos).sub(newMyWorldPos);
  // Don't push down elevated tracks (bridge, slope, bridgeCurve)
  const ttype = trackGroup.userData.type;
  if (ttype === 'bridge' || ttype === 'slope' || ttype === 'bridgeCurveLeft' || ttype === 'bridgeCurveRight') {
    _snapOffset.y = Math.max(_snapOffset.y, 0);
  }
  trackGroup.position.add(_snapOffset);
  trackGroup.updateMatrixWorld(true);

  return true;
}

function snapTrack(trackGroup) {
  const snap = findSnapTarget(trackGroup);
  if (!snap) return false;

  const myWorldConn = snap.myConn;
  const theirWorldConn = snap.theirConn;

  // Move track so that myConn aligns with theirConn
  _snapOffset.copy(theirWorldConn.pos).sub(myWorldConn.pos);
  // Don't push down elevated tracks (bridge, slope, bridgeCurve)
  const ttype = trackGroup.userData.type;
  if (ttype === 'bridge' || ttype === 'slope' || ttype === 'bridgeCurveLeft' || ttype === 'bridgeCurveRight') {
    _snapOffset.y = Math.max(_snapOffset.y, 0);
  }
  trackGroup.position.add(_snapOffset);

  // Rotate to align directions (they should be opposite)
  const myDir = snap.myConn.dir;
  const targetDirX = -theirWorldConn.dir.x;
  const targetDirZ = -theirWorldConn.dir.z;

  // Only rotate in Y
  const angle = Math.atan2(targetDirX, targetDirZ) - Math.atan2(myDir.x, myDir.z);
  if (Math.abs(angle) > 0.01) {
    // Rotate around the connection point
    _snapPivot.copy(theirWorldConn.pos);
    trackGroup.position.sub(_snapPivot);
    trackGroup.position.applyAxisAngle(_yAxis, angle);
    trackGroup.position.add(_snapPivot);
    trackGroup.rotation.y += angle;
  }

  trackGroup.updateMatrixWorld(true);
  return true;
}

// ─── Bridge Pillar Overlap Detection ───
// Hides bridge/bridgeCurve pillars (and braces) that overlap with ground-level tracks
const _pillarWorldPos = new THREE.Vector3();
const _groundTrackCenter = new THREE.Vector3();

function isGroundTrack(type) {
  return type === 'straight' || type === 'curveLeft' || type === 'curveRight' || type === 'crossing';
}

function getGroundTrackFootprint(track) {
  // Returns an array of {center, halfExtentX, halfExtentZ, angle} representing
  // the oriented bounding area of the ground track in world space
  const type = track.userData.type;
  const conns = track.userData.connectors;
  if (!conns || conns.length < 2) return [];

  // Get world positions of the two main connectors
  const p0 = conns[0].pos.clone().applyMatrix4(track.matrixWorld);
  const p1 = conns[1].pos.clone().applyMatrix4(track.matrixWorld);

  if (type === 'straight') {
    const cx = (p0.x + p1.x) / 2;
    const cz = (p0.z + p1.z) / 2;
    const dx = p1.x - p0.x;
    const dz = p1.z - p0.z;
    const trackLen = Math.sqrt(dx * dx + dz * dz);
    const angle = Math.atan2(dx, dz);
    return [{ cx, cz, halfLen: trackLen / 2 + 0.2, halfW: 0.9, angle }];
  }

  if (type === 'crossing') {
    // Crossing occupies a square area
    const cx = track.position.x;
    const cz = track.position.z;
    return [{ cx, cz, halfLen: 2.0, halfW: 2.0, angle: 0 }];
  }

  if (type === 'curveLeft' || type === 'curveRight') {
    // Sample multiple points along the curve to create a series of small footprint boxes
    const ud = track.userData;
    const sign = ud.direction === 'right' ? -1 : 1;
    const radius = ud.radius;
    const curveAngle = ud.curveAngle;
    const segments = 6;
    const footprints = [];
    for (let i = 0; i < segments; i++) {
      const t0 = i / segments;
      const t1 = (i + 1) / segments;
      const a0 = t0 * curveAngle;
      const a1 = t1 * curveAngle;
      const cp0 = curvePoint(a0, radius, sign);
      const cp1 = curvePoint(a1, radius, sign);
      // Transform to world space
      const w0 = new THREE.Vector3(cp0.x, 0, cp0.z).applyMatrix4(track.matrixWorld);
      const w1 = new THREE.Vector3(cp1.x, 0, cp1.z).applyMatrix4(track.matrixWorld);
      const cx = (w0.x + w1.x) / 2;
      const cz = (w0.z + w1.z) / 2;
      const ddx = w1.x - w0.x;
      const ddz = w1.z - w0.z;
      const segLen = Math.sqrt(ddx * ddx + ddz * ddz);
      const angle = Math.atan2(ddx, ddz);
      footprints.push({ cx, cz, halfLen: segLen / 2 + 0.2, halfW: 0.9, angle });
    }
    return footprints;
  }

  return [];
}

function pointInOrientedBox(px, pz, box) {
  // Transform point into box local space
  const dx = px - box.cx;
  const dz = pz - box.cz;
  const cosA = Math.cos(-box.angle);
  const sinA = Math.sin(-box.angle);
  const localX = dx * cosA - dz * sinA;
  const localZ = dx * sinA + dz * cosA;
  return Math.abs(localX) <= box.halfW && Math.abs(localZ) <= box.halfLen;
}

function updateBridgePillarVisibility() {
  // Collect all ground-level track footprints
  const groundFootprints = [];
  for (const track of placedTracks) {
    if (isGroundTrack(track.userData.type)) {
      groundFootprints.push(...getGroundTrackFootprint(track));
    }
  }

  // For each bridge/bridgeCurve track, check each pillar/brace against ground footprints
  for (const track of placedTracks) {
    const type = track.userData.type;
    if (type !== 'bridge' && type !== 'bridgeCurveLeft' && type !== 'bridgeCurveRight') continue;

    track.updateMatrixWorld(true);

    if (type === 'bridge') {
      // Bridge has 4 pillars: bridge_pillar_0..3
      // Pillars 0,2 are at front z, pillars 1,3 are at back z — they form 2 columns
      // Column 0 (front): pillars 0, 2  |  Column 1 (back): pillars 1, 3
      const columns = [[0, 2], [1, 3]];
      for (const col of columns) {
        // Get world position of one pillar in this column to test overlap
        const pillarChild = track.children.find(c => c.name === `bridge_pillar_${col[0]}`);
        if (!pillarChild) continue;
        _pillarWorldPos.copy(pillarChild.position).applyMatrix4(track.matrixWorld);

        const overlaps = groundFootprints.some(fp => pointInOrientedBox(_pillarWorldPos.x, _pillarWorldPos.z, fp));
        for (const pi of col) {
          const p = track.children.find(c => c.name === `bridge_pillar_${pi}`);
          if (p) p.visible = !overlaps;
        }
      }
    } else {
      // Bridge curve: pillars bcurve_pillar_${i}_l, bcurve_pillar_${i}_r and braces bcurve_brace_${i}
      const numPillars = 6;
      for (let i = 0; i < numPillars; i++) {
        const pillarL = track.children.find(c => c.name === `bcurve_pillar_${i}_l`);
        const pillarR = track.children.find(c => c.name === `bcurve_pillar_${i}_r`);
        const brace = track.children.find(c => c.name === `bcurve_brace_${i}`);

        // Use the midpoint between left and right pillar as the test point
        if (!pillarL || !pillarR) continue;
        _pillarWorldPos.copy(pillarL.position).add(pillarR.position).multiplyScalar(0.5);
        _pillarWorldPos.applyMatrix4(track.matrixWorld);

        const overlaps = groundFootprints.some(fp => pointInOrientedBox(_pillarWorldPos.x, _pillarWorldPos.z, fp));
        pillarL.visible = !overlaps;
        pillarR.visible = !overlaps;
        if (brace) brace.visible = !overlaps;
      }
    }
  }
}

function placeTrack(worldPos) {
  const track = createTrackByType(selectedTrackType);
  track.position.copy(worldPos);
  track.rotation.y = ghostRotation; // apply current ghost rotation
  track.name = `track_${selectedTrackType}_${placedTracks.length}`;
  scene.add(track);
  placedTracks.push(track);
  track.updateMatrixWorld(true);

  // Try auto-snap (with rotation), fall back to basic snap
  if (!autoSnapTrack(track)) {
    snapTrack(track);
  }

  // Rebuild train path if train is active
  if (trainGroup) updateTrainPath();

  // Update bridge pillar visibility based on overlaps with ground tracks
  updateBridgePillarVisibility();

  return track;
}

function removeTrack(track) {
  const idx = placedTracks.indexOf(track);
  if (idx >= 0) {
    placedTracks.splice(idx, 1);
    scene.remove(track);
    // Only dispose non-shared geometries to avoid breaking other tracks
    const sharedGeos = new Set([sharedTieGeo, sharedWideTieGeo, sharedPegGeo, sharedHoleGeo]);
    track.traverse(child => {
      if (child.geometry && !sharedGeos.has(child.geometry)) {
        child.geometry.dispose();
      }
    });
    // Clear references that may point to the removed track
    if (selectedPlacedTrack === track) deselectPlacedTrack();
    if (dofTrackTarget === track) { dofFocusOnTrack = false; dofTrackTarget = null; }
    // Rebuild train path if train is active
    if (trainGroup) updateTrainPath();
    // After deleting a track, pause camera follow for 3 seconds
    // so the camera doesn't immediately snap to the train
    if (trainRunning) {
      cameraFollowActive = false;
      lastCameraInteraction = performance.now();
    }
    // Update bridge pillar visibility based on overlaps with ground tracks
    updateBridgePillarVisibility();
  }
}

// ─── Metallic imperfection textures (small and fast) ───
function generateMetalImperfectionMap() {
  const size = 32;
  const { canvas, ctx } = createCanvas(size, size);
  const rng = seededRandom(4201);
  ctx.fillStyle = 'rgb(140,140,140)';
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = 'rgba(180,180,180,0.35)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 10; i++) {
    ctx.beginPath();
    const x0 = rng() * size, y0 = rng() * size;
    const ang = rng() * Math.PI * 2;
    const len = 4 + rng() * 12;
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + Math.cos(ang) * len, y0 + Math.sin(ang) * len);
    ctx.stroke();
  }
  for (let i = 0; i < 6; i++) {
    ctx.fillStyle = `rgba(60,60,60,${0.15 + rng() * 0.2})`;
    ctx.beginPath();
    ctx.arc(rng() * size, rng() * size, 1 + rng() * 2, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

function generateMetalNormalMap() {
  const size = 32;
  const { canvas, ctx } = createCanvas(size, size);
  ctx.fillStyle = 'rgb(128,128,255)';
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

const metalImperfectionMap = generateMetalImperfectionMap();
const metalNormalMap = generateMetalNormalMap();

function createMetalMaterial(color, roughness = 0.35, metalness = 0.85) {
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(color),
    roughness,
    metalness,
    roughnessMap: metalImperfectionMap,
    normalMap: metalNormalMap,
    normalScale: new THREE.Vector2(0.4, 0.4),
    envMapIntensity: 1.2,
  });
}

// ─── Train Creation ───
// Wheel references for rolling animation
let trainWheelMeshes = [];
let trainWheelRadii = [];
let trainPrevPos = new THREE.Vector3();
let trainCurrentYaw = 0; // for smooth rotation
let trainCurrentPitch = 0; // for slope tilt

// Wagon references
let wagonGroups = [];
let wagonWheelMeshes = []; // array of arrays
let wagonWheelRadii = [];
let wagonCurrentYaws = [];
let wagonCurrentPitches = [];
const WAGON_GAP = 7.0; // path-parameter gap between wagons (each index ≈ 0.25 world units)
const NUM_WAGONS = 2;

// Helper quaternions for composing yaw + pitch correctly
const _qYaw = new THREE.Quaternion();
const _qPitch = new THREE.Quaternion();
const _yawAxis = new THREE.Vector3(0, 1, 0);
const _xAxis = new THREE.Vector3(1, 0, 0);
function applyYawPitch(obj, yaw, pitch) {
  _qYaw.setFromAxisAngle(_yawAxis, yaw);
  _qPitch.setFromAxisAngle(_xAxis, pitch);
  obj.quaternion.copy(_qYaw).multiply(_qPitch);
}

// Dynamic coupling bars between train<->wagon and wagon<->wagon
let couplingBars = []; // array of { mesh, bar, hookA, hookB }

function createCouplingBar() {
  const group = new THREE.Group();
  group.name = 'coupling_bar_group';

  // Main coupling rod
  const barMat = new THREE.MeshStandardMaterial({ color: '#C8A035', roughness: 0.3, metalness: 0.5 });
  const barGeo = new THREE.CylinderGeometry(0.04, 0.04, 1, 8);
  barGeo.rotateX(Math.PI / 2); // align along Z
  const bar = new THREE.Mesh(barGeo, barMat);
  bar.castShadow = true;
  bar.name = 'coupling_rod';
  group.add(bar);

  // Hook ring at front end (connects to rear of leading vehicle)
  const hookGeo = new THREE.TorusGeometry(0.06, 0.018, 8, 12);
  const hookMat = new THREE.MeshStandardMaterial({ color: '#8B7355', roughness: 0.35, metalness: 0.5 });
  const hookA = new THREE.Mesh(hookGeo, hookMat);
  hookA.rotation.y = Math.PI / 2;
  hookA.name = 'coupling_hookA';
  group.add(hookA);

  // Hook ring at rear end (connects to front of trailing vehicle)
  const hookB = new THREE.Mesh(hookGeo, hookMat);
  hookB.rotation.y = Math.PI / 2;
  hookB.name = 'coupling_hookB';
  group.add(hookB);

  return { group, bar, hookA, hookB };
}

function updateCouplingBar(coupling, rearWorldPos, frontWorldPos, yHeight) {
  const { group, bar, hookA, hookB } = coupling;

  // Midpoint
  const mx = (rearWorldPos.x + frontWorldPos.x) / 2;
  const mz = (rearWorldPos.z + frontWorldPos.z) / 2;
  group.position.set(mx, yHeight, mz);

  // Direction and distance
  const dx = frontWorldPos.x - rearWorldPos.x;
  const dz = frontWorldPos.z - rearWorldPos.z;
  const dist = Math.sqrt(dx * dx + dz * dz);
  const angle = Math.atan2(dx, dz);

  group.rotation.y = angle;

  // Scale bar length to span the gap
  const barLen = Math.max(0.05, dist - 0.12); // leave room for hooks
  bar.scale.z = barLen;
  bar.position.set(0, 0, 0);

  // Position hooks at each end
  hookA.position.set(0, 0, -dist / 2 + 0.02);
  hookB.position.set(0, 0, dist / 2 - 0.02);
}

// Shared wheel detail materials (created once, reused across all train & wagon wheels)
const _wheelDarkMat = new THREE.MeshStandardMaterial({ color: '#4A4A4A', roughness: 0.25, metalness: 0.7 });
const _wheelMetalMat = new THREE.MeshStandardMaterial({ color: '#888888', roughness: 0.3, metalness: 0.8 });
const _wheelSpokeMat = new THREE.MeshStandardMaterial({ color: '#C85028', roughness: 0.35, metalness: 0.4 });

function createTrain() {
  const train = new THREE.Group();
  train.name = 'train';
  // Rotation will be applied via quaternion for correct slope tilt in all directions
  trainWheelMeshes = [];
  trainWheelRadii = [];

  // --- Vintage toy locomotive materials (matching reference image) ---
  const bodyMat = createWoodMaterial('#171B12', 0.30, 'wagonBed');      // dark olive body
  const cabinMat = createWoodMaterial('#B8242C', 0.32, 'wagonWall');   // matching flatbed wagon red (no wood overlay)
  const roofMat = createWoodMaterial('#1A1A1A', 0.28, 'black');         // black roof
  const stackMat = createWoodMaterial('#3A3A3A', 0.30, 'gold');         // dark iron stack
  const stackCapMat = createWoodMaterial('#C8A035', 0.22, 'goldCap');   // gold cap
  const wheelMat = createWoodMaterial('#B8242C', 0.30, 'redWheel');     // red wheels (matching flatbed wagon red)
  const tanMat = createWoodMaterial('#C8A035', 0.25, 'brass');          // brass/gold trim
  const chassisMat = createWoodMaterial('#111111', 0.35, 'chassis');    // black chassis/base
  const ironMat = createWoodMaterial('#4A4A4A', 0.40, 'iron');          // dark iron for details
  const hubMat = createWoodMaterial('#C8A035', 0.25, 'hub');            // brass hub caps

  // Store references for UI bindings
  trainMats.body = bodyMat;
  trainMats.cabin = cabinMat;
  trainMats.roof = roofMat;
  trainMats.stack = stackMat;
  trainMats.stackCap = stackCapMat;
  trainMats.wheel = wheelMat;
  trainMats.trim = tanMat;

  // --- Reference heights ---
  const RAIL_TOP = TRACK_BED_H + TRACK_RAIL_H;
  const BIG_WHEEL_R = 0.24;    // larger rear drive wheels
  const SMALL_WHEEL_R = 0.16;  // smaller front wheels
  const WHEEL_THICK = 0.08;
  const WHEEL_CY_BIG = RAIL_TOP + BIG_WHEEL_R;
  const WHEEL_CY_SMALL = RAIL_TOP + SMALL_WHEEL_R;

  // Body proportions — vintage locomotive style
  const BODY_RADIUS = 0.32;
  const BODY_LEN = 1.30;
  const CHASSIS_H = 0.16;
  const CHASSIS_W = 0.80;
  const CHASSIS_BOTTOM = WHEEL_CY_BIG - BIG_WHEEL_R * 0.3;
  const CHASSIS_Y = CHASSIS_BOTTOM + CHASSIS_H / 2;

  // Wheel Z positions
  const FRONT_WHEEL_Z = -0.42;
  const REAR_WHEEL_Z = 0.22;

  // --- Black chassis/base plate ---
  const chassisGeo = new RoundedBoxGeometry(CHASSIS_W, CHASSIS_H, BODY_LEN + 0.40, 2, 0.03);
  const chassis = new THREE.Mesh(chassisGeo, chassisMat);
  chassis.position.set(0, CHASSIS_Y, 0);
  chassis.castShadow = true;
  chassis.name = 'train_chassis';
  train.add(chassis);

  // --- Main boiler (cylinder, horizontal) ---
  const BOILER_RADIUS = BODY_RADIUS * 0.88;
  const BOILER_CY = CHASSIS_BOTTOM + CHASSIS_H + BOILER_RADIUS + 0.02;
  const BOILER_LEN = BODY_LEN * 0.72;
  const boilerGeo = new THREE.CylinderGeometry(BOILER_RADIUS, BOILER_RADIUS, BOILER_LEN, 20);
  const boiler = new THREE.Mesh(boilerGeo, bodyMat);
  boiler.rotation.x = Math.PI / 2;
  boiler.position.set(0, BOILER_CY, -0.12);
  boiler.castShadow = true;
  boiler.name = 'train_boiler';
  train.add(boiler);

  // Boiler front cap (sphere section)
  const boilerCapGeo = new THREE.SphereGeometry(BOILER_RADIUS, 20, 14, 0, Math.PI * 2, 0, Math.PI / 2);
  const boilerCap = new THREE.Mesh(boilerCapGeo, bodyMat);
  boilerCap.rotation.x = Math.PI / 2;
  boilerCap.position.set(0, BOILER_CY, -0.12 - BOILER_LEN / 2);
  boilerCap.castShadow = true;
  boilerCap.name = 'train_boiler_cap';
  train.add(boilerCap);

  // --- Large front face disc (concentric rings like in image) ---
  const FRONT_FACE_Z = -0.12 - BOILER_LEN / 2 - 0.01;
  // Outer ring
  const frontRingGeo = new THREE.TorusGeometry(BOILER_RADIUS * 0.75, 0.025, 10, 24);
  const frontRing = new THREE.Mesh(frontRingGeo, ironMat);
  frontRing.position.set(0, BOILER_CY, FRONT_FACE_Z);
  frontRing.name = 'train_front_ring';
  train.add(frontRing);
  // Middle ring
  const frontRing2Geo = new THREE.TorusGeometry(BOILER_RADIUS * 0.50, 0.02, 8, 20);
  const frontRing2 = new THREE.Mesh(frontRing2Geo, ironMat);
  frontRing2.position.set(0, BOILER_CY, FRONT_FACE_Z - 0.01);
  frontRing2.name = 'train_front_ring2';
  train.add(frontRing2);
  // Inner ring
  const frontRing3Geo = new THREE.TorusGeometry(BOILER_RADIUS * 0.25, 0.015, 8, 16);
  const frontRing3 = new THREE.Mesh(frontRing3Geo, ironMat);
  frontRing3.position.set(0, BOILER_CY, FRONT_FACE_Z - 0.02);
  frontRing3.name = 'train_front_ring3';
  train.add(frontRing3);
  // Center hub (brass) with embedded light
  const frontHubGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.04, 12);
  const _frontHubMat = new THREE.MeshStandardMaterial({
    color: '#C8A035', emissive: '#FFEE88', emissiveIntensity: 0.4, roughness: 0.2, metalness: 0.3
  });
  trainFrontHubMat = _frontHubMat;
  const frontHub = new THREE.Mesh(frontHubGeo, _frontHubMat);
  frontHub.rotation.x = Math.PI / 2;
  frontHub.position.set(0, BOILER_CY, FRONT_FACE_Z - 0.02);
  frontHub.name = 'train_front_hub';
  train.add(frontHub);

  // Front hub point light (warm glow from the yellow cylinder)
  const _frontHubLight = new THREE.PointLight(0xFFEE88, 0, 4, 2);
  _frontHubLight.position.set(0, BOILER_CY, FRONT_FACE_Z - 0.05);
  _frontHubLight.name = 'train_front_hub_light';
  train.add(_frontHubLight);
  trainFrontHubLight = _frontHubLight;

  // --- Cabin (red rear block with gold trim, like image) ---
  const CABIN_W = CHASSIS_W - 0.02;
  const CABIN_H = 0.54;
  const CABIN_D = 0.52;
  const CABIN_Z = BODY_LEN / 2 - CABIN_D / 2 + 0.08;
  const CABIN_CY = CHASSIS_BOTTOM + CHASSIS_H + CABIN_H / 2;
  const cabinGeo = new RoundedBoxGeometry(CABIN_W, CABIN_H, CABIN_D, 3, 0.04);
  const cabin = new THREE.Mesh(cabinGeo, cabinMat);
  cabin.position.set(0, CABIN_CY, CABIN_Z);
  cabin.castShadow = true;
  cabin.name = 'train_cabin';
  train.add(cabin);

  // Cabin gold trim strip (horizontal band around cabin — upper)
  const cabinTrimGeo = new RoundedBoxGeometry(CABIN_W + 0.02, 0.030, CABIN_D + 0.02, 1, 0.005);
  const cabinTrim = new THREE.Mesh(cabinTrimGeo, tanMat);
  cabinTrim.position.set(0, CABIN_CY + 0.12, CABIN_Z);
  cabinTrim.name = 'train_cabin_trim';
  train.add(cabinTrim);

  // Cabin gold trim strip (lower)
  const cabinTrimLo = new THREE.Mesh(cabinTrimGeo, tanMat);
  cabinTrimLo.position.set(0, CABIN_CY - 0.12, CABIN_Z);
  cabinTrimLo.name = 'train_cabin_trim_lo';
  train.add(cabinTrimLo);

  // Red inset panel on cabin sides (like image)
  const panelInsetGeo = new RoundedBoxGeometry(0.014, 0.16, 0.32, 1, 0.01);
  const panelInsetMat = createWoodMaterial('#991C1C', 0.32, 'cabinPanel');
  for (const side of [-1, 1]) {
    const p = new THREE.Mesh(panelInsetGeo, panelInsetMat);
    p.position.set(side * (CABIN_W / 2 + 0.004), CABIN_CY, CABIN_Z);
    p.name = `train_cabin_panel_${side > 0 ? 'r' : 'l'}`;
    train.add(p);
  }

  // Cabin roof (arched dark roof)
  const cabinRoofGeo = new RoundedBoxGeometry(CABIN_W + 0.08, 0.06, CABIN_D + 0.12, 2, 0.03);
  const cabinRoof = new THREE.Mesh(cabinRoofGeo, roofMat);
  cabinRoof.position.set(0, CABIN_CY + CABIN_H / 2 + 0.03, CABIN_Z);
  cabinRoof.castShadow = true;
  cabinRoof.name = 'train_cabin_roof';
  train.add(cabinRoof);

  // Cabin windows (dark insets, arched style)
  const windowMat = new THREE.MeshStandardMaterial({ color: '#0A1520', emissive: '#000000', emissiveIntensity: 0, roughness: 0.1, metalness: 0.3 });
  trainCabinWindowMats = [windowMat];
  const sideWinGeo = new RoundedBoxGeometry(0.012, 0.18, 0.22, 1, 0.02);
  for (const side of [-1, 1]) {
    const win = new THREE.Mesh(sideWinGeo, windowMat);
    win.position.set(side * (CABIN_W / 2 + 0.005), CABIN_CY + 0.10, CABIN_Z);
    win.name = `train_window_${side > 0 ? 'r' : 'l'}`;
    train.add(win);
  }
  // Gold window frames
  const winFrameGeo = new RoundedBoxGeometry(0.016, 0.22, 0.26, 1, 0.01);
  for (const side of [-1, 1]) {
    const frame = new THREE.Mesh(winFrameGeo, tanMat);
    frame.position.set(side * (CABIN_W / 2 + 0.006), CABIN_CY + 0.10, CABIN_Z);
    frame.name = `train_winframe_${side > 0 ? 'r' : 'l'}`;
    train.add(frame);
  }

  // Circular side porthole on cabin (like image)
  const portholeFrameGeo = new THREE.TorusGeometry(0.07, 0.012, 8, 16);
  const portholeGlassGeo = new THREE.CircleGeometry(0.06, 16);
  for (const side of [-1, 1]) {
    const pf = new THREE.Mesh(portholeFrameGeo, tanMat);
    pf.rotation.y = Math.PI / 2;
    pf.position.set(side * (CABIN_W / 2 + 0.006), CABIN_CY + 0.08, CABIN_Z - 0.02);
    pf.name = `train_porthole_frame_${side > 0 ? 'r' : 'l'}`;
    train.add(pf);
    const pg = new THREE.Mesh(portholeGlassGeo, windowMat);
    pg.rotation.y = side > 0 ? -Math.PI / 2 : Math.PI / 2;
    pg.position.set(side * (CABIN_W / 2 + 0.005), CABIN_CY + 0.08, CABIN_Z - 0.02);
    pg.name = `train_porthole_glass_${side > 0 ? 'r' : 'l'}`;
    train.add(pg);
  }

  const rearWinGeo = new RoundedBoxGeometry(0.20, 0.18, 0.012, 1, 0.02);
  const rearWin = new THREE.Mesh(rearWinGeo, windowMat);
  rearWin.position.set(0, CABIN_CY + 0.10, CABIN_Z + CABIN_D / 2 + 0.003);
  rearWin.name = 'train_window_rear';
  train.add(rearWin);

  // --- Smokestack (tall dark iron funnel with wide flared top, like image) ---
  const STACK_Z = -BODY_LEN * 0.30;
  const STACK_BASE_Y = BOILER_CY + BOILER_RADIUS;
  // Narrow base cylinder
  const stackBaseGeo = new THREE.CylinderGeometry(0.05, 0.07, 0.12, 14);
  const stackBase = new THREE.Mesh(stackBaseGeo, stackMat);
  stackBase.position.set(0, STACK_BASE_Y + 0.06, STACK_Z);
  stackBase.castShadow = true;
  stackBase.name = 'train_stack_base';
  train.add(stackBase);
  // Main stack tube (slightly tapered)
  const stackGeo = new THREE.CylinderGeometry(0.06, 0.055, 0.28, 14);
  const stack = new THREE.Mesh(stackGeo, stackMat);
  stack.position.set(0, STACK_BASE_Y + 0.26, STACK_Z);
  stack.castShadow = true;
  stack.name = 'train_stack_tube';
  train.add(stack);
  // Wide flared funnel cap (like the image — big inverted cone)
  const stackCapGeo = new THREE.CylinderGeometry(0.14, 0.065, 0.10, 16);
  const stackCap = new THREE.Mesh(stackCapGeo, stackCapMat);
  stackCap.position.set(0, STACK_BASE_Y + 0.45, STACK_Z);
  stackCap.castShadow = true;
  stackCap.name = 'train_stack_cap';
  train.add(stackCap);
  // Stack rim ring
  const stackRimGeo = new THREE.TorusGeometry(0.14, 0.012, 8, 20);
  const stackRim = new THREE.Mesh(stackRimGeo, tanMat);
  stackRim.rotation.x = Math.PI / 2;
  stackRim.position.set(0, STACK_BASE_Y + 0.50, STACK_Z);
  stackRim.name = 'train_stack_rim';
  train.add(stackRim);

  // --- Steam dome (brass dome on top of boiler) ---
  const domeGeo = new THREE.SphereGeometry(0.10, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2);
  const dome = new THREE.Mesh(domeGeo, hubMat);
  dome.position.set(0, BOILER_CY + BOILER_RADIUS, 0.05);
  dome.castShadow = true;
  dome.name = 'train_dome';
  train.add(dome);
  // Dome base ring
  const domeBaseGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.03, 12);
  const domeBase = new THREE.Mesh(domeBaseGeo, stackCapMat);
  domeBase.position.set(0, BOILER_CY + BOILER_RADIUS + 0.015, 0.05);
  domeBase.name = 'train_dome_base';
  train.add(domeBase);

  // --- Second smaller dome / sand dome (like image has two domes) ---
  const dome2Geo = new THREE.SphereGeometry(0.07, 10, 6, 0, Math.PI * 2, 0, Math.PI / 2);
  const dome2 = new THREE.Mesh(dome2Geo, hubMat);
  dome2.position.set(0, BOILER_CY + BOILER_RADIUS, -0.20);
  dome2.castShadow = true;
  dome2.name = 'train_dome2';
  train.add(dome2);
  const dome2BaseGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.025, 10);
  const dome2Base = new THREE.Mesh(dome2BaseGeo, stackCapMat);
  dome2Base.position.set(0, BOILER_CY + BOILER_RADIUS + 0.012, -0.20);
  dome2Base.name = 'train_dome2_base';
  train.add(dome2Base);

  // --- Front bumper / cowcatcher (slanted guard with grille bars like image) ---
  const cowW = CHASSIS_W + 0.04;
  const cowH = 0.14;
  const cowD = 0.22;
  const cowGeo = new RoundedBoxGeometry(cowW, cowH, cowD, 2, 0.02);
  const cow = new THREE.Mesh(cowGeo, bodyMat);
  cow.position.set(0, CHASSIS_Y - 0.02, -BODY_LEN / 2 - cowD / 2 + 0.02);
  cow.castShadow = true;
  cow.name = 'train_cowcatcher';
  train.add(cow);

  // Cowcatcher grille bars (vertical slats)
  for (let i = -3; i <= 3; i++) {
    const barGeo = new RoundedBoxGeometry(0.022, cowH + 0.02, cowD + 0.08, 1, 0.004);
    const bar = new THREE.Mesh(barGeo, bodyMat);
    bar.position.set(i * 0.09, CHASSIS_Y - 0.04, -BODY_LEN / 2 - cowD / 2 - 0.02);
    bar.name = `train_cowbar_${i + 3}`;
    train.add(bar);
  }
  // Cowcatcher bottom angled plate
  const cowBottomGeo = new RoundedBoxGeometry(cowW - 0.10, 0.04, cowD + 0.10, 1, 0.01);
  const cowBottom = new THREE.Mesh(cowBottomGeo, chassisMat);
  cowBottom.position.set(0, CHASSIS_Y - cowH / 2 - 0.02, -BODY_LEN / 2 - cowD / 2 - 0.03);
  cowBottom.name = 'train_cow_bottom';
  train.add(cowBottom);

  // --- Gold trim strips along boiler sides ---
  const stripeGeo = new RoundedBoxGeometry(0.018, 0.03, BOILER_LEN - 0.06, 1, 0.005);
  for (const side of [-1, 1]) {
    const stripe = new THREE.Mesh(stripeGeo, tanMat);
    stripe.position.set(side * (BOILER_RADIUS - 0.02), BOILER_CY, -0.12);
    stripe.name = `train_stripe_${side > 0 ? 'r' : 'l'}`;
    train.add(stripe);
  }

  // --- Headlight removed (lamp lens + spotlight) ---
  trainHeadlightLensMat = null;
  trainHeadlight = null;
  trainHeadlightTarget = null;

  // Cabin interior glow
  const _cabinLight = new THREE.PointLight(0xFFCC66, 0, 3, 2);
  _cabinLight.position.set(0, CABIN_CY, CABIN_Z);
  _cabinLight.name = 'train_cabin_light';
  train.add(_cabinLight);
  trainCabinLight = _cabinLight;

  // Rear tail light
  const rearLightMat2 = new THREE.MeshStandardMaterial({
    color: '#FF2200', emissive: '#FF2200', emissiveIntensity: 0, roughness: 0.2, metalness: 0.1
  });
  const _rearLight = new THREE.PointLight(0xFF3300, 0, 2, 2);
  _rearLight.position.set(0, CABIN_CY - 0.08, CABIN_Z + CABIN_D / 2 + 0.04);
  _rearLight.name = 'train_rear_light';
  train.add(_rearLight);
  trainRearLight = _rearLight;
  trainRearLight._meshMat = rearLightMat2;

  // --- Rear coupling anchor ---
  const rearCouplingMat = new THREE.MeshStandardMaterial({ color: '#C8A035', roughness: 0.3, metalness: 0.4 });
  const rearMountGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.08, 10);
  const rearMount = new THREE.Mesh(rearMountGeo, rearCouplingMat);
  rearMount.rotation.x = Math.PI / 2;
  rearMount.position.set(0, CHASSIS_Y, BODY_LEN / 2 + 0.08);
  rearMount.castShadow = true;
  rearMount.name = 'train_rear_mount';
  train.add(rearMount);

  const rearHookGeo = new THREE.TorusGeometry(0.055, 0.016, 8, 12);
  const rearHook = new THREE.Mesh(rearHookGeo, rearCouplingMat);
  rearHook.rotation.y = Math.PI / 2;
  rearHook.position.set(0, CHASSIS_Y, BODY_LEN / 2 + 0.13);
  rearHook.name = 'train_rear_hook';
  train.add(rearHook);

  train._rearCoupleZ = BODY_LEN / 2 + 0.13;
  train._coupleY = CHASSIS_Y;

  // --- Model-train style disc wheels (concentric rings, flange, center nub) ---
  function createDiscWheel(radius, thick, mat, axisMat) {
    const grp = new THREE.Group();
    const darkMat = _wheelDarkMat;
    const metalMat = _wheelMetalMat;

    // Outer flange rim – slightly larger than disc, thin
    const flangeR = radius * 1.12;
    const flangeThick = thick * 0.25;
    const flangeGeo = new THREE.CylinderGeometry(flangeR, flangeR, flangeThick, 24);
    const flange = new THREE.Mesh(flangeGeo, darkMat);
    flange.rotation.z = Math.PI / 2;
    flange.position.x = -thick / 2 + flangeThick / 2 - 0.005; // inner side
    grp.add(flange);

    // Main disc body
    const discGeo = new THREE.CylinderGeometry(radius, radius, thick, 24);
    const disc = new THREE.Mesh(discGeo, darkMat);
    disc.rotation.z = Math.PI / 2;
    grp.add(disc);

    // Concentric ring grooves on the outer face (lathe-turned look)
    const ring1R = radius * 0.82;
    const ring1Geo = new THREE.TorusGeometry(ring1R, 0.008, 6, 28);
    const ring1 = new THREE.Mesh(ring1Geo, darkMat);
    ring1.rotation.y = Math.PI / 2;
    ring1.position.x = thick / 2 + 0.001;
    grp.add(ring1);

    const ring2R = radius * 0.58;
    const ring2Geo = new THREE.TorusGeometry(ring2R, 0.007, 6, 24);
    const ring2 = new THREE.Mesh(ring2Geo, darkMat);
    ring2.rotation.y = Math.PI / 2;
    ring2.position.x = thick / 2 + 0.001;
    grp.add(ring2);

    const ring3R = radius * 0.36;
    const ring3Geo = new THREE.TorusGeometry(ring3R, 0.006, 6, 20);
    const ring3 = new THREE.Mesh(ring3Geo, darkMat);
    ring3.rotation.y = Math.PI / 2;
    ring3.position.x = thick / 2 + 0.001;
    grp.add(ring3);

    // Center hub (raised cylindrical boss)
    const hubR = radius * 0.18;
    const hubThick = thick * 0.6;
    const hubGeo = new THREE.CylinderGeometry(hubR, hubR, hubThick, 14);
    const hub = new THREE.Mesh(hubGeo, darkMat);
    hub.rotation.z = Math.PI / 2;
    hub.position.x = thick / 2 + hubThick / 2 - 0.005;
    grp.add(hub);

    // Center axle nub (small metallic pin sticking out)
    const nubR = radius * 0.06;
    const nubLen = thick * 0.5;
    const nubGeo = new THREE.CylinderGeometry(nubR, nubR, nubLen, 8);
    const nub = new THREE.Mesh(nubGeo, metalMat);
    nub.rotation.z = Math.PI / 2;
    nub.position.x = thick / 2 + hubThick + nubLen / 2 - 0.01;
    grp.add(nub);

    // Spoke bars radiating from hub to near rim
    const NUM_SPOKES = 8;
    const spokeInnerR = hubR * 1.1;
    const spokeOuterR = radius * 0.88;
    const spokeLen = spokeOuterR - spokeInnerR;
    const spokeBarW = radius * 0.06;
    const spokeBarH = thick * 0.3;
    const spokeGeo = new THREE.BoxGeometry(spokeBarW, spokeLen, spokeBarH);
    for (let i = 0; i < NUM_SPOKES; i++) {
      const angle = (i / NUM_SPOKES) * Math.PI * 2;
      const midR = (spokeInnerR + spokeOuterR) / 2;
      const spoke = new THREE.Mesh(spokeGeo, _wheelSpokeMat);
      spoke.position.set(
        thick / 2 + 0.003,
        Math.cos(angle) * midR,
        Math.sin(angle) * midR
      );
      spoke.rotation.x = angle;
      grp.add(spoke);
    }

    return grp;
  }

  // Axle bar connecting wheel pairs
  const axleBarMat = new THREE.MeshStandardMaterial({ color: '#999999', roughness: 0.3, metalness: 0.8 });

  // Big rear drive wheels
  let wi = 0;
  const bigAxleLen = CHASSIS_W + WHEEL_THICK;
  const bigAxleGeo = new THREE.CylinderGeometry(0.012, 0.012, bigAxleLen, 6);
  const bigAxle = new THREE.Mesh(bigAxleGeo, axleBarMat);
  bigAxle.rotation.z = Math.PI / 2;
  bigAxle.position.set(0, WHEEL_CY_BIG, REAR_WHEEL_Z);
  bigAxle.name = 'train_rear_axle';
  train.add(bigAxle);

  for (const side of [-1, 1]) {
    const wg = new THREE.Group();
    wg.position.set(side * (CHASSIS_W / 2 + WHEEL_THICK / 2), WHEEL_CY_BIG, REAR_WHEEL_Z);
    wg.name = `train_wheelgrp_${wi}`;
    const discWheel = createDiscWheel(BIG_WHEEL_R, WHEEL_THICK, wheelMat, hubMat);
    // Mirror so flange faces outward
    if (side === -1) discWheel.scale.x = -1;
    discWheel.name = `train_wheel_${wi}`;
    discWheel.castShadow = true;
    wg.add(discWheel);
    train.add(wg);
    trainWheelMeshes.push(wg);
    trainWheelRadii.push(BIG_WHEEL_R);
    wi++;
  }
  // Smaller front wheels
  const smallAxleLen = CHASSIS_W + WHEEL_THICK;
  const smallAxleGeo = new THREE.CylinderGeometry(0.010, 0.010, smallAxleLen, 6);
  const smallAxle = new THREE.Mesh(smallAxleGeo, axleBarMat);
  smallAxle.rotation.z = Math.PI / 2;
  smallAxle.position.set(0, WHEEL_CY_SMALL, FRONT_WHEEL_Z);
  smallAxle.name = 'train_front_axle';
  train.add(smallAxle);

  for (const side of [-1, 1]) {
    const wg = new THREE.Group();
    wg.position.set(side * (CHASSIS_W / 2 + WHEEL_THICK / 2), WHEEL_CY_SMALL, FRONT_WHEEL_Z);
    wg.name = `train_wheelgrp_${wi}`;
    const discWheel = createDiscWheel(SMALL_WHEEL_R, WHEEL_THICK, wheelMat, hubMat);
    if (side === -1) discWheel.scale.x = -1;
    discWheel.name = `train_wheel_${wi}`;
    discWheel.castShadow = true;
    wg.add(discWheel);
    train.add(wg);
    trainWheelMeshes.push(wg);
    trainWheelRadii.push(SMALL_WHEEL_R);
    wi++;
  }

  // --- Roof-mounted lantern light on cabin ---
  const CABIN_ROOF_TOP = CABIN_CY + CABIN_H / 2 + 0.06;
  const lanternBaseMat = new THREE.MeshStandardMaterial({ color: '#2A2A2A', roughness: 0.4, metalness: 0.6 });
  const lanternBaseGeo = new THREE.CylinderGeometry(0.04, 0.05, 0.04, 10);
  const lanternBase = new THREE.Mesh(lanternBaseGeo, lanternBaseMat);
  lanternBase.position.set(0, CABIN_ROOF_TOP + 0.02, CABIN_Z);
  lanternBase.name = 'train_lantern_base';
  train.add(lanternBase);

  const lanternGlassGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.07, 10);
  const lanternGlassMat = new THREE.MeshStandardMaterial({
    color: '#FFFFAA', emissive: '#FFEE66', emissiveIntensity: 0.5,
    roughness: 0.1, metalness: 0.0, transparent: true, opacity: 0.85
  });
  const lanternGlass = new THREE.Mesh(lanternGlassGeo, lanternGlassMat);
  lanternGlass.position.set(0, CABIN_ROOF_TOP + 0.075, CABIN_Z);
  lanternGlass.name = 'train_lantern_glass';
  train.add(lanternGlass);

  const lanternCapGeo = new THREE.ConeGeometry(0.045, 0.04, 10);
  const lanternCap = new THREE.Mesh(lanternCapGeo, lanternBaseMat);
  lanternCap.position.set(0, CABIN_ROOF_TOP + 0.13, CABIN_Z);
  lanternCap.name = 'train_lantern_cap';
  train.add(lanternCap);

  // Lantern point light (warm glow)
  const lanternLight = new THREE.PointLight(0xFFDD88, 0.6, 4, 2);
  lanternLight.position.set(0, CABIN_ROOF_TOP + 0.08, CABIN_Z);
  lanternLight.name = 'train_lantern_light';
  train.add(lanternLight);

  return train;
}

// ─── Wagon Creation ───
function createWagon(index) {
  const wagon = new THREE.Group();
  wagon.name = `wagon_${index}`;
  // Rotation will be applied via quaternion for correct slope tilt in all directions

  const RAIL_TOP = TRACK_BED_H + TRACK_RAIL_H;
  const WHEEL_R = 0.16;
  const WHEEL_THICK = 0.10;
  const WHEEL_CY = RAIL_TOP + WHEEL_R;

  const CHASSIS_H = 0.12;
  const CHASSIS_W = 0.756;
  const CHASSIS_LEN = 1.30;
  const CHASSIS_BOTTOM = WHEEL_CY - WHEEL_R * 0.3;
  const CHASSIS_Y = CHASSIS_BOTTOM + CHASSIS_H / 2;

  const chassisMat = createWoodMaterial('#111111', 0.35, 'wagonChassis');

  // Chassis
  const chassisGeo = new RoundedBoxGeometry(CHASSIS_W, CHASSIS_H, CHASSIS_LEN, 2, 0.02);
  const chassis = new THREE.Mesh(chassisGeo, chassisMat);
  chassis.position.set(0, CHASSIS_Y, 0);
  chassis.castShadow = true;
  chassis.name = `wagon_${index}_chassis`;
  wagon.add(chassis);

  // Alternate wagon types: cargo box / flatbed with barrel
  const isCargoWagon = index === 0;

  if (isCargoWagon) {
    // Cargo box wagon — green body with gold trim (matches cabin style)
    const boxH = 0.38;
    const boxW = CHASSIS_W - 0.06;
    const boxD = CHASSIS_LEN - 0.12;
    const boxY = CHASSIS_BOTTOM + CHASSIS_H + boxH / 2;
    const boxMat = createWoodMaterial('#1A5C3A', 0.32, 'wagonBox');
    const boxGeo = new RoundedBoxGeometry(boxW, boxH, boxD, 2, 0.04);
    const box = new THREE.Mesh(boxGeo, boxMat);
    box.position.set(0, boxY, 0);
    box.castShadow = true;
    box.name = `wagon_${index}_box`;
    wagon.add(box);

    // Gold trim horizontal band
    const trimGeo = new RoundedBoxGeometry(boxW + 0.02, 0.03, boxD + 0.02, 1, 0.005);
    const trimMat = createWoodMaterial('#C8A035', 0.25, 'wagonTrim');
    const trim = new THREE.Mesh(trimGeo, trimMat);
    trim.position.set(0, boxY - 0.02, 0);
    trim.name = `wagon_${index}_trim`;
    wagon.add(trim);

    // Gold letters / panel detail (simplified as gold rectangles on sides)
    const panelGeo = new RoundedBoxGeometry(0.012, 0.12, 0.50, 1, 0.005);
    for (const side of [-1, 1]) {
      const panel = new THREE.Mesh(panelGeo, trimMat);
      panel.position.set(side * (boxW / 2 + 0.004), boxY + 0.04, 0);
      panel.name = `wagon_${index}_panel_${side > 0 ? 'r' : 'l'}`;
      wagon.add(panel);
    }
  } else {
    // Flatbed wagon with red body and barrels/crates
    const bedH = 0.10;
    const bedW = CHASSIS_W - 0.04;
    const bedD = CHASSIS_LEN - 0.10;
    const bedY = CHASSIS_BOTTOM + CHASSIS_H + bedH / 2;
    const bedMat = createWoodMaterial('#B8242C', 0.30, 'wagonBed');
    const bedGeo = new RoundedBoxGeometry(bedW, bedH, bedD, 2, 0.02);
    const bed = new THREE.Mesh(bedGeo, bedMat);
    bed.position.set(0, bedY, 0);
    bed.castShadow = true;
    bed.name = `wagon_${index}_bed`;
    wagon.add(bed);

    // Low side walls
    const wallH = 0.20;
    const wallGeo = new RoundedBoxGeometry(0.04, wallH, bedD, 1, 0.01);
    const wallMat = createWoodMaterial('#B8242C', 0.32, 'wagonWall');
    for (const side of [-1, 1]) {
      const wall = new THREE.Mesh(wallGeo, wallMat);
      wall.position.set(side * (bedW / 2 - 0.02), bedY + bedH / 2 + wallH / 2, 0);
      wall.castShadow = true;
      wall.name = `wagon_${index}_wall_${side > 0 ? 'r' : 'l'}`;
      wagon.add(wall);
    }
    // End walls
    const endWallGeo = new RoundedBoxGeometry(bedW, wallH, 0.04, 1, 0.01);
    for (const end of [-1, 1]) {
      const ew = new THREE.Mesh(endWallGeo, wallMat);
      ew.position.set(0, bedY + bedH / 2 + wallH / 2, end * (bedD / 2 - 0.02));
      ew.castShadow = true;
      ew.name = `wagon_${index}_endwall_${end > 0 ? 'b' : 'f'}`;
      wagon.add(ew);
    }

    // Barrels on the flatbed
    const barrelMat = createWoodMaterial('#8B6D3F', 0.38, 'barrel');
    const barrelGeo = new THREE.CylinderGeometry(0.10, 0.10, 0.22, 10);
    for (let b = 0; b < 3; b++) {
      const barrel = new THREE.Mesh(barrelGeo, barrelMat);
      barrel.position.set((b - 1) * 0.26, bedY + bedH / 2 + 0.11, -0.15);
      barrel.castShadow = true;
      barrel.name = `wagon_${index}_barrel_${b}`;
      wagon.add(barrel);
    }
    // Crate
    const crateMat = createWoodMaterial('#C8A87A', 0.35, 'crate');
    const crateGeo = new RoundedBoxGeometry(0.24, 0.20, 0.24, 1, 0.02);
    const crate = new THREE.Mesh(crateGeo, crateMat);
    crate.position.set(0, bedY + bedH / 2 + 0.10, 0.25);
    crate.castShadow = true;
    crate.name = `wagon_${index}_crate`;
    wagon.add(crate);
  }

  // --- Coupling anchors (small mount stubs + hook loops at each end) ---
  const hookMat = createWoodMaterial('#C8A035', 0.28, 'hook');
  const mountGeo = new THREE.CylinderGeometry(0.055, 0.055, 0.07, 10);
  const loopGeo = new THREE.TorusGeometry(0.05, 0.015, 8, 12);
  for (const end of [-1, 1]) {
    const mount = new THREE.Mesh(mountGeo, hookMat);
    mount.rotation.x = Math.PI / 2;
    mount.position.set(0, CHASSIS_Y, end * (CHASSIS_LEN / 2 + 0.05));
    mount.castShadow = true;
    mount.name = `wagon_${index}_mount_${end > 0 ? 'b' : 'f'}`;
    wagon.add(mount);

    const loop = new THREE.Mesh(loopGeo, hookMat);
    loop.rotation.y = Math.PI / 2;
    loop.position.set(0, CHASSIS_Y, end * (CHASSIS_LEN / 2 + 0.10));
    loop.name = `wagon_${index}_loop_${end > 0 ? 'b' : 'f'}`;
    wagon.add(loop);
  }

  // Store coupling offsets for dynamic bar positioning
  wagon._frontCoupleZ = -(CHASSIS_LEN / 2 + 0.10);
  wagon._rearCoupleZ = (CHASSIS_LEN / 2 + 0.10);
  wagon._coupleY = CHASSIS_Y;

  // --- Model-train disc wheels (matching locomotive style) ---
  function createWagonDiscWheel(radius, thick) {
    const grp = new THREE.Group();
    const darkMat = _wheelDarkMat;
    const metalMat = _wheelMetalMat;

    // Outer flange
    const flangeR = radius * 1.12;
    const flangeThick = thick * 0.25;
    const flangeGeo = new THREE.CylinderGeometry(flangeR, flangeR, flangeThick, 24);
    const flange = new THREE.Mesh(flangeGeo, darkMat);
    flange.rotation.z = Math.PI / 2;
    flange.position.x = -thick / 2 + flangeThick / 2 - 0.005;
    grp.add(flange);

    // Main disc
    const discGeo = new THREE.CylinderGeometry(radius, radius, thick, 24);
    const disc = new THREE.Mesh(discGeo, darkMat);
    disc.rotation.z = Math.PI / 2;
    grp.add(disc);

    // Concentric rings
    const ring1Geo = new THREE.TorusGeometry(radius * 0.78, 0.006, 6, 24);
    const ring1 = new THREE.Mesh(ring1Geo, darkMat);
    ring1.rotation.y = Math.PI / 2;
    ring1.position.x = thick / 2 + 0.001;
    grp.add(ring1);

    const ring2Geo = new THREE.TorusGeometry(radius * 0.52, 0.005, 6, 20);
    const ring2 = new THREE.Mesh(ring2Geo, darkMat);
    ring2.rotation.y = Math.PI / 2;
    ring2.position.x = thick / 2 + 0.001;
    grp.add(ring2);

    // Hub boss
    const hubR = radius * 0.2;
    const hubThick = thick * 0.55;
    const hubGeo = new THREE.CylinderGeometry(hubR, hubR, hubThick, 12);
    const hub = new THREE.Mesh(hubGeo, darkMat);
    hub.rotation.z = Math.PI / 2;
    hub.position.x = thick / 2 + hubThick / 2 - 0.005;
    grp.add(hub);

    // Axle nub
    const nubGeo = new THREE.CylinderGeometry(radius * 0.06, radius * 0.06, thick * 0.4, 8);
    const nub = new THREE.Mesh(nubGeo, metalMat);
    nub.rotation.z = Math.PI / 2;
    nub.position.x = thick / 2 + hubThick + thick * 0.15;
    grp.add(nub);

    // Spoke bars radiating from hub to near rim
    const spokeMat = new THREE.MeshStandardMaterial({ color: '#C85028', roughness: 0.35, metalness: 0.4 });
    const NUM_SPOKES = 8;
    const spokeInnerR = hubR * 1.1;
    const spokeOuterR = radius * 0.85;
    const spokeLen = spokeOuterR - spokeInnerR;
    const spokeBarW = radius * 0.06;
    const spokeBarH = thick * 0.3;
    const spokeGeo = new THREE.BoxGeometry(spokeBarW, spokeLen, spokeBarH);
    for (let i = 0; i < NUM_SPOKES; i++) {
      const angle = (i / NUM_SPOKES) * Math.PI * 2;
      const midR = (spokeInnerR + spokeOuterR) / 2;
      const spoke = new THREE.Mesh(spokeGeo, spokeMat);
      spoke.position.set(
        thick / 2 + 0.003,
        Math.cos(angle) * midR,
        Math.sin(angle) * midR
      );
      spoke.rotation.x = angle;
      grp.add(spoke);
    }

    return grp;
  }

  const axleBarMat = new THREE.MeshStandardMaterial({ color: '#999999', roughness: 0.3, metalness: 0.8 });
  const wheelPositionsZ = [-0.38, 0.38];
  const wheels = [];
  let wi = 0;

  // Add axle bars connecting each pair
  for (const zOff of wheelPositionsZ) {
    const axleLen = CHASSIS_W + WHEEL_THICK;
    const axleGeo = new THREE.CylinderGeometry(0.010, 0.010, axleLen, 6);
    const axle = new THREE.Mesh(axleGeo, axleBarMat);
    axle.rotation.z = Math.PI / 2;
    axle.position.set(0, WHEEL_CY, zOff);
    axle.name = `wagon_${index}_axle_${zOff > 0 ? 'rear' : 'front'}`;
    wagon.add(axle);
  }

  for (const side of [-1, 1]) {
    for (const zOff of wheelPositionsZ) {
      const wg = new THREE.Group();
      wg.position.set(side * (CHASSIS_W / 2 + WHEEL_THICK / 2 - 0.02), WHEEL_CY, zOff);
      wg.name = `wagon_${index}_wheelgrp_${wi}`;

      const discWheel = createWagonDiscWheel(WHEEL_R, WHEEL_THICK);
      if (side === -1) discWheel.scale.x = -1;
      discWheel.castShadow = true;
      discWheel.name = `wagon_${index}_wheel_${wi}`;
      wg.add(discWheel);

      wagon.add(wg);
      wheels.push({ group: wg, radius: WHEEL_R });
      wi++;
    }
  }
  wagon._wheels = wheels;

  return wagon;
}

// ─── Track Path Extraction ───
function buildTrackPath() {
  if (placedTracks.length === 0) return [];

  const points = [];
  const visited = new Set();
  const SAMPLE_STEP = 0.25;

  // Try to build a connected chain starting from first track
  let current = placedTracks[0];
  let connectorIdx = 0; // start from connector 0
  visited.add(current);

  let _isFirstPiece = true;

  function sampleTrackPiece(track, fromConnIdx) {
    const ud = track.userData;
    const conns = ud.connectors;
    if (!conns || conns.length < 2) return;

    // Determine traversal direction
    const type = ud.type;
    const entering = fromConnIdx;
    let exiting;
    if (type === 'crossing') {
      // Crossing pairs: 0↔1 (Z-axis), 2↔3 (X-axis)
      const crossPair = { 0: 1, 1: 0, 2: 3, 3: 2 };
      exiting = crossPair[entering] !== undefined ? crossPair[entering] : (entering === 0 ? conns.length - 1 : 0);
    } else {
      exiting = entering === 0 ? conns.length - 1 : 0;
    }

    // Skip first sample point on subsequent pieces to avoid duplicate junction points
    // (the previous piece's last point is the same as this piece's first point)
    const startI = _isFirstPiece ? 0 : 1;
    _isFirstPiece = false;
    if (type === 'straight' || type === 'bridge' || type === 'slope' || type === 'crossing') {
      const p0 = conns[entering].pos.clone().applyMatrix4(track.matrixWorld);
      const p1 = conns[exiting].pos.clone().applyMatrix4(track.matrixWorld);
      const dist = p0.distanceTo(p1);
      const steps = Math.max(2, Math.ceil(dist / SAMPLE_STEP));
      for (let i = startI; i <= steps; i++) {
        const t = i / steps;
        points.push(new THREE.Vector3(
          p0.x + (p1.x - p0.x) * t,
          p0.y + (p1.y - p0.y) * t,
          p0.z + (p1.z - p0.z) * t
        ));
      }
    } else if (type === 'curveLeft' || type === 'curveRight') {
      const radius = ud.radius;
      const angle = ud.curveAngle;
      const dir = ud.direction;
      const sign = dir === 'right' ? -1 : 1;
      const steps = Math.max(8, Math.ceil((radius * angle) / SAMPLE_STEP));

      for (let i = startI; i <= steps; i++) {
        let t = i / steps;
        if (entering !== 0) t = 1 - t;
        const a = t * angle;
        const local = curvePoint(a, radius, sign);
        points.push(local.clone().applyMatrix4(track.matrixWorld));
      }
    } else if (type === 'bridgeCurveLeft' || type === 'bridgeCurveRight') {
      const radius = ud.radius;
      const angle = ud.curveAngle;
      const dir = ud.direction;
      const sign = dir === 'right' ? -1 : 1;
      const bHeight = ud.height || 1.2;
      const steps = Math.max(8, Math.ceil((radius * angle) / SAMPLE_STEP));

      for (let i = startI; i <= steps; i++) {
        let t = i / steps;
        if (entering !== 0) t = 1 - t;
        const a = t * angle;
        const local = curvePoint(a, radius, sign);
        local.y = bHeight;
        points.push(local.clone().applyMatrix4(track.matrixWorld));
      }
    }
  }

  // Build chain
  sampleTrackPiece(current, 0);

  // Follow connections
  for (let iter = 0; iter < placedTracks.length; iter++) {
    const currentConns = getWorldConnectors(current);
    let exitIdx;
    if (current.userData.type === 'crossing') {
      const crossPair = { 0: 1, 1: 0, 2: 3, 3: 2 };
      exitIdx = crossPair[connectorIdx] !== undefined ? crossPair[connectorIdx] : (connectorIdx === 0 ? currentConns.length - 1 : 0);
    } else {
      exitIdx = connectorIdx === 0 ? currentConns.length - 1 : 0;
    }
    const exitConn = currentConns[exitIdx];

    let found = false;
    for (const other of placedTracks) {
      if (visited.has(other)) continue;
      const otherConns = getWorldConnectors(other);
      for (let ci = 0; ci < otherConns.length; ci++) {
        const dist = exitConn.pos.distanceTo(otherConns[ci].pos);
        const dot = exitConn.dir.dot(otherConns[ci].dir);
        if (dist < 0.5 && dot < -0.3) {
          visited.add(other);
          sampleTrackPiece(other, ci);
          current = other;
          connectorIdx = ci;
          found = true;
          break;
        }
      }
      if (found) break;
    }
    if (!found) break;
  }

  // Re-parameterize by arc length for uniform speed
  if (points.length < 2) return points;

  // Compute cumulative distances
  const cumDist = [0];
  for (let i = 1; i < points.length; i++) {
    cumDist.push(cumDist[i - 1] + points[i].distanceTo(points[i - 1]));
  }
  const totalLength = cumDist[cumDist.length - 1];
  if (totalLength < 0.001) return points;

  // Resample at uniform intervals
  const uniformStep = SAMPLE_STEP;
  const numSamples = Math.max(2, Math.ceil(totalLength / uniformStep));
  const resampled = [];
  let srcIdx = 0;
  for (let i = 0; i <= numSamples; i++) {
    const targetDist = (i / numSamples) * totalLength;
    while (srcIdx < cumDist.length - 2 && cumDist[srcIdx + 1] < targetDist) srcIdx++;
    const segLen = cumDist[srcIdx + 1] - cumDist[srcIdx];
    const frac = segLen > 0.0001 ? (targetDist - cumDist[srcIdx]) / segLen : 0;
    resampled.push(new THREE.Vector3().lerpVectors(points[srcIdx], points[Math.min(srcIdx + 1, points.length - 1)], frac));
  }

  return resampled;
}

function updateTrainPath() {
  const prevLen = trainPathPoints.length;
  trainPathPoints = buildTrackPath();
  updatePathLoopFlag();
  // If the train was stopped at end of track, check if it can resume
  if (trainStopped && trainGroup) {
    const newLen = trainPathPoints.length - 1;
    if (_pathIsLoop || newLen > trainT + 1) {
      // Path extended or became a loop — resume the train
      trainStopped = false;
      trainRunning = true;
      trainPaused = false;
      updateTrainBtnUI();
    }
  }
}

const _trainPos = new THREE.Vector3();
const _trainLookDir = new THREE.Vector3();
const _camOffset = new THREE.Vector3();
const _prevTrainPos = new THREE.Vector3();
const _targetCamPos = new THREE.Vector3();
let cameraFollowInit = false;

// ─── Camera follow / orbit interaction ───
let cameraFollowActive = true;
let lastCameraInteraction = 0;
const CAMERA_RESUME_DELAY = 3000; // ms before auto-resuming follow

// ─── DOF focus target tracking ───
let dofFocusOnTrack = false;       // true when user is interacting with a track
let dofTrackTarget = null;          // the track group currently focused
let lastTrackInteraction = 0;       // timestamp of last track click/drag
const DOF_TRACK_RESUME_DELAY = 3000; // ms before DOF returns to train
const _dofTrackWorldPos = new THREE.Vector3(); // reusable for track world position

// ─── DOF zoom-to-train tracking ───
let dofZoomFocusTrain = false;   // true when zoom triggers DOF refocus on train
let lastZoomTime = 0;            // timestamp of last zoom event
const DOF_ZOOM_SETTLE_DELAY = 1500; // ms after last zoom before DOF stops adjusting

// ─── DOF zoom-with-selection tracking ───
let dofZoomingWithSelection = false; // true when zooming while something is selected
let lastSelectionZoomTime = 0;       // timestamp of last zoom while selected

function pauseCameraFollow() {
  if (trainRunning) {
    cameraFollowActive = false;
    lastCameraInteraction = performance.now();
  }
}

renderer.domElement.addEventListener('pointerdown', pauseCameraFollow);
renderer.domElement.addEventListener('wheel', (e) => {
  pauseCameraFollow();
  if (dofFocusOnTrack && dofTrackTarget) {
    // Keep DOF focused on the selected object while zooming
    lastTrackInteraction = performance.now();
    dofZoomingWithSelection = true;
    lastSelectionZoomTime = performance.now();
  } else if (trainGroup) {
    // When zooming with nothing selected, trigger DOF refocus on the train
    dofZoomFocusTrain = true;
    lastZoomTime = performance.now();
  }
});

// Reusable vectors for wagon positioning
const _wagonPos = new THREE.Vector3();
const _wagonLookDir = new THREE.Vector3();
const _wagonPrevPositions = [];
const _cRear = new THREE.Vector3();
const _cFront = new THREE.Vector3();

// Cache loop detection per path update to avoid per-frame recalculation
let _pathIsLoop = false;
function updatePathLoopFlag() {
  if (trainPathPoints.length < 2) { _pathIsLoop = false; return; }
  _pathIsLoop = trainPathPoints[0].distanceTo(trainPathPoints[trainPathPoints.length - 1]) < 1.0;
}

// Compute the initial yaw (and pitch) for a vehicle at path parameter t
function getInitialOrientation(t, pathPoints) {
  const lookAheadT = 5;
  const { pos } = getPathPosition(t, pathPoints);
  const lookPos = getPathPositionAhead(t, lookAheadT, pathPoints);
  const dx = lookPos.x - pos.x;
  const dy = lookPos.y - pos.y;
  const dz = lookPos.z - pos.z;
  const horizLen = Math.sqrt(dx * dx + dz * dz);
  let yaw = 0, pitch = 0;
  if (horizLen > 0.0001) {
    yaw = Math.atan2(dx, dz) + Math.PI;
    pitch = Math.atan2(dy, horizLen);
  }
  return { yaw, pitch };
}

const _gpPosResult = new THREE.Vector3(); // reusable to avoid per-frame allocation
function getPathPosition(t, pathPoints) {
  const len = pathPoints.length - 1;
  let wt = t;
  if (_pathIsLoop) {
    wt = ((wt % len) + len) % len;
  } else {
    wt = Math.max(0, Math.min(wt, len));
  }
  const idx = Math.floor(wt);
  const frac = wt - idx;
  const i0 = Math.min(idx, len);
  const i1 = _pathIsLoop ? ((idx + 1) % pathPoints.length) : Math.min(idx + 1, len);
  _gpPosResult.lerpVectors(pathPoints[i0], pathPoints[i1], frac);
  return { pos: _gpPosResult, i0, i1 };
}

const _gpAheadResult = new THREE.Vector3(); // reusable to avoid per-frame allocation
function getPathPositionAhead(t, ahead, pathPoints) {
  const len = pathPoints.length - 1;
  let wt = t + ahead;
  if (_pathIsLoop) {
    wt = ((wt % len) + len) % len;
  } else {
    wt = Math.max(0, Math.min(wt, len));
  }
  const idx = Math.floor(wt);
  const frac = wt - idx;
  const i0 = Math.min(idx, len);
  const i1 = _pathIsLoop ? ((idx + 1) % pathPoints.length) : Math.min(idx + 1, len);
  _gpAheadResult.lerpVectors(pathPoints[i0], pathPoints[i1], frac);
  return _gpAheadResult;
}

const _fullLookDir = new THREE.Vector3(); // reusable for look direction
const _wFullLookDir = new THREE.Vector3(); // reusable for wagon look direction

function moveTrainAlongPath(dt) {
  if (!trainGroup || trainPathPoints.length < 2) return;
  if (trainStopped) return;

  const len = trainPathPoints.length - 1;
  trainT += TRAIN_SPEED * dt;
  if (_pathIsLoop) {
    trainT = ((trainT % len) + len) % len;
  } else if (trainT >= len) {
    // Reached end of disconnected track — stop the train
    trainT = len;
    trainStopped = true;
    trainRunning = false;
    trainPaused = false;
    updateTrainBtnUI();
    return;
  }

  const { pos: tPos } = getPathPosition(trainT, trainPathPoints);
  _trainPos.copy(tPos);

  // Compute distance traveled for wheel rolling
  const distMoved = _trainPos.distanceTo(trainPrevPos);
  // Guard against teleport-sized jumps (e.g. loop wrap) causing stalls or spikes
  const safeDist = distMoved > 2.0 ? 0 : distMoved;
  trainPrevPos.copy(_trainPos);

  trainGroup.position.copy(_trainPos);

  // Look direction — sample a point ahead on the path parameter (not raw index)
  const lookAheadT = 5;
  const lookPos = getPathPositionAhead(trainT, lookAheadT, trainPathPoints);
  _fullLookDir.set(lookPos.x - _trainPos.x, lookPos.y - _trainPos.y, lookPos.z - _trainPos.z);
  _trainLookDir.set(_fullLookDir.x, 0, _fullLookDir.z);
  if (_trainLookDir.lengthSq() > 0.0001) {
    const horizLen = _trainLookDir.length();
    _trainLookDir.normalize();
    const targetYaw = Math.atan2(_trainLookDir.x, _trainLookDir.z) + Math.PI;

    // Compute target pitch (tilt on slopes)
    const targetPitch = Math.atan2(_fullLookDir.y, horizLen);

    // Smooth rotation using angular lerp
    const smoothFactor = 1.0 - Math.pow(0.02, dt / 60);

    let diffYaw = targetYaw - trainCurrentYaw;
    while (diffYaw > Math.PI) diffYaw -= Math.PI * 2;
    while (diffYaw < -Math.PI) diffYaw += Math.PI * 2;
    trainCurrentYaw += diffYaw * smoothFactor;

    let diffPitch = targetPitch - trainCurrentPitch;
    while (diffPitch > Math.PI) diffPitch -= Math.PI * 2;
    while (diffPitch < -Math.PI) diffPitch += Math.PI * 2;
    trainCurrentPitch += diffPitch * smoothFactor;

    applyYawPitch(trainGroup, trainCurrentYaw, trainCurrentPitch);
  }

  // Roll train wheels based on distance traveled
  for (let w = 0; w < trainWheelMeshes.length; w++) {
    const wg = trainWheelMeshes[w];
    const r = trainWheelRadii[w];
    const angularDelta = safeDist / r;
    wg.rotation.x -= angularDelta;
  }

  // --- Move wagons along the path behind the train ---
  for (let wIdx = 0; wIdx < wagonGroups.length; wIdx++) {
    const wg = wagonGroups[wIdx];
    if (!wg) continue;
    const wagonT = trainT - WAGON_GAP * (wIdx + 1);
    const { pos: wPos } = getPathPosition(wagonT, trainPathPoints);
    _wagonPos.copy(wPos);

    // Initialize prev position tracking
    if (!_wagonPrevPositions[wIdx]) _wagonPrevPositions[wIdx] = _wagonPos.clone();
    const wDistMoved = _wagonPos.distanceTo(_wagonPrevPositions[wIdx]);
    const wSafeDist = wDistMoved > 2.0 ? 0 : wDistMoved;
    _wagonPrevPositions[wIdx].copy(_wagonPos);

    wg.position.copy(_wagonPos);

    // Wagon look direction — use path parameter ahead
    const wLookPos = getPathPositionAhead(wagonT, lookAheadT, trainPathPoints);
    _wFullLookDir.set(wLookPos.x - _wagonPos.x, wLookPos.y - _wagonPos.y, wLookPos.z - _wagonPos.z);
    _wagonLookDir.set(_wFullLookDir.x, 0, _wFullLookDir.z);
    if (_wagonLookDir.lengthSq() > 0.0001) {
      const wHorizLen = _wagonLookDir.length();
      _wagonLookDir.normalize();
      const targetYaw = Math.atan2(_wagonLookDir.x, _wagonLookDir.z) + Math.PI;
      const targetPitch = Math.atan2(_wFullLookDir.y, wHorizLen);

      const smoothFactor = 1.0 - Math.pow(0.02, dt / 60);

      let diffYaw = targetYaw - wagonCurrentYaws[wIdx];
      while (diffYaw > Math.PI) diffYaw -= Math.PI * 2;
      while (diffYaw < -Math.PI) diffYaw += Math.PI * 2;
      wagonCurrentYaws[wIdx] += diffYaw * smoothFactor;

      if (!wagonCurrentPitches[wIdx]) wagonCurrentPitches[wIdx] = 0;
      let diffPitch = targetPitch - wagonCurrentPitches[wIdx];
      while (diffPitch > Math.PI) diffPitch -= Math.PI * 2;
      while (diffPitch < -Math.PI) diffPitch += Math.PI * 2;
      wagonCurrentPitches[wIdx] += diffPitch * smoothFactor;

      applyYawPitch(wg, wagonCurrentYaws[wIdx], wagonCurrentPitches[wIdx]);
    }

    // Roll wagon wheels
    if (wagonWheelMeshes[wIdx]) {
      for (const ww of wagonWheelMeshes[wIdx]) {
        const angDelta = wSafeDist / ww.radius;
        ww.group.rotation.x -= angDelta;
      }
    }
  }

  // --- Update dynamic coupling bars between vehicles ---
  for (let ci = 0; ci < couplingBars.length; ci++) {
    const cb = couplingBars[ci];
    if (!cb) continue;

    // Leading vehicle: train (ci===0) or wagon[ci-1]
    let leader, leaderRearZ, leaderCoupleY;
    if (ci === 0) {
      leader = trainGroup;
      leaderRearZ = trainGroup._rearCoupleZ || 0.73;
      leaderCoupleY = trainGroup._coupleY || 0.28;
    } else {
      leader = wagonGroups[ci - 1];
      if (!leader) continue;
      leaderRearZ = leader._rearCoupleZ || 0.75;
      leaderCoupleY = leader._coupleY || 0.28;
    }

    // Trailing vehicle: wagon[ci]
    const trailer = wagonGroups[ci];
    if (!trailer) continue;
    const trailerFrontZ = trailer._frontCoupleZ || -0.75;
    const trailerCoupleY = trailer._coupleY || 0.28;

    // Get world positions of coupling points
    _cRear.set(0, leaderCoupleY, leaderRearZ);
    leader.localToWorld(_cRear);

    _cFront.set(0, trailerCoupleY, trailerFrontZ);
    trailer.localToWorld(_cFront);

    const avgY = (_cRear.y + _cFront.y) / 2;
    updateCouplingBar(cb, _cRear, _cFront, avgY);
  }
}

// ─── Ghost preview ───
// Cache ghost tracks to avoid re-creating on every mouse move
let ghostTrackType = '';

function updateGhost(worldPos) {
  // Only recreate ghost if type changed
  if (ghostTrack && ghostTrackType === selectedTrackType) {
    ghostTrack.position.copy(worldPos);
    ghostTrack.rotation.y = ghostRotation;
    ghostTrack.updateMatrixWorld(true);
    // Auto-snap ghost to nearby connectors
    autoSnapTrack(ghostTrack);
    return;
  }

  if (ghostTrack) {
    scene.remove(ghostTrack);
    ghostTrack.traverse(child => {
      if (child.geometry) child.geometry.dispose();
    });
  }
  ghostTrack = createTrackByType(selectedTrackType);
  ghostTrackType = selectedTrackType;
  ghostTrack.position.copy(worldPos);
  ghostTrack.rotation.y = ghostRotation;
  ghostTrack.name = 'ghostTrack';
  ghostTrack.traverse(child => {
    if (child.isMesh || child.isInstancedMesh) {
      const oldColor = child.material.color ? child.material.color.clone() : new THREE.Color(0x888888);
      child.material = new THREE.MeshBasicMaterial({
        color: oldColor,
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
      });
      child.castShadow = false;
      child.receiveShadow = false;
    }
  });
  scene.add(ghostTrack);
  ghostTrack.updateMatrixWorld(true);
  // Auto-snap ghost to nearby connectors
  autoSnapTrack(ghostTrack);
}

function removeGhost() {
  if (ghostTrack) {
    scene.remove(ghostTrack);
    ghostTrack.traverse(child => {
      if (child.geometry) child.geometry.dispose();
    });
    ghostTrack = null;
    ghostTrackType = '';
  }
}

// ─── Track hover highlight for deletion / rotation ───
const _highlightMatCache = new WeakMap(); // cache tinted materials to avoid re-cloning
const _rotateHighlightMatCache = new WeakMap(); // cache for rotate-tool tinted materials
const _selectHighlightMatCache = new WeakMap(); // cache for selection-tool tinted materials
const rotateHighlightColor = new THREE.Color(0x44aaff);
const selectHighlightColor = new THREE.Color(0xff6655);
let _lastHighlightMode = null; // 'delete', 'rotate', or 'select'

function highlightTrack(track, forceMode) {
  const mode = forceMode || (selectionTool ? 'select' : (rotateTool ? 'rotate' : 'delete'));
  if (hoveredTrack === track && _lastHighlightMode === mode) return;
  unhighlightTrack();
  hoveredTrack = track;
  _lastHighlightMode = mode;
  const isRotate = mode === 'rotate';
  const isSelect = mode === 'select';
  let tintColor, cache, emissiveColor;
  if (isSelect) {
    tintColor = selectHighlightColor;
    cache = _selectHighlightMatCache;
    emissiveColor = 0xcc3322;
  } else if (isRotate) {
    tintColor = rotateHighlightColor;
    cache = _rotateHighlightMatCache;
    emissiveColor = 0x2266ff;
  } else {
    tintColor = highlightColor;
    cache = _highlightMatCache;
    emissiveColor = 0xff2222;
  }
  track.traverse(child => {
    if ((child.isMesh || child.isInstancedMesh) && child.material) {
      if (!originalMaterials.has(child)) {
        originalMaterials.set(child, child.material);
      }
      const origMat = originalMaterials.get(child);
      // Reuse cached tinted material if available
      let tintMat = cache.get(origMat);
      if (!tintMat) {
        tintMat = origMat.clone();
        if (tintMat.color) tintMat.color.lerp(tintColor, 0.35);
        if (tintMat.emissive) {
          tintMat.emissive.set(emissiveColor);
          tintMat.emissiveIntensity = 0.12;
        }
        cache.set(origMat, tintMat);
      }
      child.material = tintMat;
    }
  });
}

function unhighlightTrack() {
  if (!hoveredTrack) return;
  // If this track is the selected track, restore the selection highlight instead of original
  if (hoveredTrack === selectedPlacedTrack) {
    applySelectionHighlight(hoveredTrack);
  } else {
    hoveredTrack.traverse(child => {
      if ((child.isMesh || child.isInstancedMesh) && originalMaterials.has(child)) {
        child.material = originalMaterials.get(child);
      }
    });
  }
  hoveredTrack = null;
}

// ─── Input Handling ───
const _groundHit = new THREE.Vector3(); // reusable for ground intersection
const _ghostSnapPos = new THREE.Vector3(); // reusable for ghost snapping
const _intersectResults = []; // reusable array for raycasting

function _updateMouseFromEvent(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function getGroundIntersect(event) {
  _updateMouseFromEvent(event);
  raycaster.setFromCamera(mouse, camera);
  raycaster.ray.intersectPlane(groundPlane, _groundHit);
  return _groundHit;
}

function getTrackUnderMouse(event) {
  _updateMouseFromEvent(event);
  raycaster.setFromCamera(mouse, camera);

  // Single batch intersect against all placed tracks — much faster than per-track calls
  _intersectResults.length = 0;
  raycaster.intersectObjects(placedTracks, true, _intersectResults);
  if (_intersectResults.length > 0) {
    // Walk up to find the top-level track group
    let obj = _intersectResults[0].object;
    while (obj && obj.parent && !placedTracks.includes(obj)) {
      obj = obj.parent;
    }
    return placedTracks.includes(obj) ? obj : null;
  }
  return null;
}

const _sceneryResults = []; // reusable array for scenery raycasting
function getSceneryUnderMouse(event) {
  _updateMouseFromEvent(event);
  raycaster.setFromCamera(mouse, camera);
  _sceneryResults.length = 0;
  // Also check the train
  const targets = [sceneryGroup];
  if (trainGroup) targets.push(trainGroup);
  raycaster.intersectObjects(targets, true, _sceneryResults);
  if (_sceneryResults.length > 0) {
    return _sceneryResults[0].object;
  }
  return null;
}

let pointerDownPos = new THREE.Vector2();
let pointerDownTime = 0;
let isOverUI = false;
let isOrbiting = false; // true when user is rotating/panning camera (no track drag)

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (isOverUI) return;
  pointerDownPos.set(e.clientX, e.clientY);
  pointerDownTime = Date.now();

  const track = getTrackUnderMouse(e);
  if (track && e.button === 0 && !rotateTool) {
    // Start drag (allowed for both placement tools and selection tool)
    isDragging = true;
    dragTrack = track;
    controls.enabled = false;
    const hit = getGroundIntersect(e);
    dragOffset.set(track.position.x - hit.x, track.position.y - hit.y, track.position.z - hit.z);
    // Focus DOF on this track
    dofFocusOnTrack = true;
    dofTrackTarget = track;
    lastTrackInteraction = performance.now();
  } else if (!track) {
    // No track hit — user is about to orbit/pan
    isOrbiting = true;
  }
});

let lastMoveTime = 0;
renderer.domElement.addEventListener('pointermove', (e) => {
  if (isOverUI) {
    removeGhost();
    unhighlightTrack();
    return;
  }

  // Dragging always gets full-speed updates
  if (isDragging && dragTrack) {
    renderer.domElement.style.cursor = 'grabbing';
    const hit = getGroundIntersect(e);
    if (!hit) return;
    // Move track to cursor position (with offset)
    dragTrack.position.set(hit.x + dragOffset.x, hit.y + dragOffset.y, hit.z + dragOffset.z);
    clampToBasePlate(dragTrack.position);
    dragTrack.updateMatrixWorld(true);
    // Try auto-snap (moves and rotates to align with nearby connectors)
    autoSnapTrack(dragTrack);
    // Keep DOF tracking the dragged track
    lastTrackInteraction = performance.now();
    return;
  }

  // Skip expensive raycasting while user is orbiting/panning the camera
  if (isOrbiting) return;

  // Throttle ghost updates to ~30fps
  const now = performance.now();
  if (now - lastMoveTime < 33) return;
  lastMoveTime = now;

  const hit = getGroundIntersect(e);
  if (!hit) return;

  // Fast path: when a track placement tool is active, skip expensive track-under-mouse
  // raycast entirely and go straight to ghost positioning
  if (selectedTrackType) {
    unhighlightTrack();
    renderer.domElement.style.cursor = '';
    const snappedPos = _ghostSnapPos;
    snappedPos.set(Math.round(hit.x * 2) / 2, 0, Math.round(hit.z * 2) / 2);
    if (!isOnBasePlate(snappedPos)) { removeGhost(); return; }
    updateGhost(snappedPos);
    // Focus DOF on the ghost track as it follows the mouse
    if (ghostTrack) {
      dofFocusOnTrack = true;
      dofTrackTarget = ghostTrack;
      lastTrackInteraction = performance.now();
    }
    return;
  }

  // Update ghost & hover highlight (only when NO track tool is active)
  const trackUnder = getTrackUnderMouse(e);
  if (!trackUnder) {
    unhighlightTrack();
    renderer.domElement.style.cursor = '';
    removeGhost();
  } else {
    removeGhost();
    highlightTrack(trackUnder);
    renderer.domElement.style.cursor = selectionTool ? 'grab' : '';
  }
});

renderer.domElement.addEventListener('pointerup', (e) => {
  isOrbiting = false;

  // Check if it was a click (not a drag)
  const dx = e.clientX - pointerDownPos.x;
  const dy = e.clientY - pointerDownPos.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const elapsed = Date.now() - pointerDownTime;
  const wasClick = dist < 5 && elapsed < 300 && e.button === 0;

  if (isDragging && dragTrack) {
    renderer.domElement.style.cursor = '';
    if (wasClick && selectionTool) {
      // Short click with selection tool — treat as select, not drag
      // Restore track to original position (snap handles it)
      snapTrack(dragTrack);
      if (trainGroup) updateTrainPath();
      const clickedTrack = dragTrack;
      isDragging = false;
      dragTrack = null;
      controls.enabled = true;
      // Toggle selection
      if (selectedPlacedTrack === clickedTrack) {
        deselectPlacedTrack();
      } else {
        selectPlacedTrack(clickedTrack);
      }
      dofFocusOnTrack = true;
      dofTrackTarget = clickedTrack;
      lastTrackInteraction = performance.now();
      return;
    }
    // Real drag — auto-snap (with rotation) and finalize
    if (!autoSnapTrack(dragTrack)) {
      snapTrack(dragTrack);
    }
    // Rebuild train path after moving a track
    if (trainGroup) updateTrainPath();
    // Keep DOF focused on the track that was just placed
    dofFocusOnTrack = true;
    dofTrackTarget = dragTrack;
    lastTrackInteraction = performance.now();
    // If selection tool, auto-select the dragged track
    if (selectionTool && dragTrack) {
      selectPlacedTrack(dragTrack);
    }
    isDragging = false;
    dragTrack = null;
    controls.enabled = true;
    return;
  }

  isDragging = false;
  dragTrack = null;
  controls.enabled = true;

  if (isOverUI) return;

  if (wasClick) {
    const track = getTrackUnderMouse(e);
    if (track && rotateTool) {
      // Rotate tool active — rotate the track 90° around its position
      const trackType = track.userData.type;
      if (trackType === 'straight' || trackType === 'curveLeft' || trackType === 'curveRight' || trackType === 'bridge' || trackType === 'slope' || trackType === 'bridgeCurveLeft' || trackType === 'bridgeCurveRight') {
        track.rotation.y += Math.PI / 2;
        track.updateMatrixWorld(true);
        // Re-snap after rotation
        snapTrack(track);
        if (trainGroup) updateTrainPath();
        updateBridgePillarVisibility();
      }
      // Focus DOF on rotated track
      dofFocusOnTrack = true;
      dofTrackTarget = track;
      lastTrackInteraction = performance.now();
    } else if (track && selectionTool) {
      // Selection tool active — toggle selection (click again to deselect)
      if (selectedPlacedTrack === track) {
        deselectPlacedTrack();
      } else {
        selectPlacedTrack(track);
      }
      dofFocusOnTrack = true;
      dofTrackTarget = track;
      lastTrackInteraction = performance.now();
    } else if (track) {
      // Clicked an existing track — focus DOF on it
      dofFocusOnTrack = true;
      dofTrackTarget = track;
      lastTrackInteraction = performance.now();
    } else {
      // Deselect any selected track when clicking empty space
      deselectPlacedTrack();
      // Check scenery objects (trees, signs, benches, lamps, train)
      const sceneryObj = getSceneryUnderMouse(e);
      if (sceneryObj) {
        dofFocusOnTrack = true;
        dofTrackTarget = sceneryObj;
        lastTrackInteraction = performance.now();
      } else {
        const hit = getGroundIntersect(e);
        if (hit) {
          // If ghost is visible and auto-snapped, use its position/rotation
          let placePos, placeRot;
          if (ghostTrack) {
            placePos = ghostTrack.position.clone();
            placeRot = ghostTrack.rotation.y;
          } else {
            placePos = new THREE.Vector3(
              Math.round(hit.x * 2) / 2,
              0,
              Math.round(hit.z * 2) / 2
            );
            placeRot = ghostRotation;
          }
          if (!isOnBasePlate(placePos) || !selectedTrackType) return;
          const savedGhostRotation = ghostRotation;
          ghostRotation = placeRot;
          const newTrack = placeTrack(placePos);
          ghostRotation = savedGhostRotation;
          updateCounter();
          // Focus DOF on newly placed track
          dofFocusOnTrack = true;
          dofTrackTarget = newTrack;
          lastTrackInteraction = performance.now();
        }
      }
    }
  }
});

// Suppress right-click context menu
renderer.domElement.addEventListener('contextmenu', (e) => {
  e.preventDefault();
});

// ─── Keyboard controls ───
window.addEventListener('keydown', (e) => {
  // R / Shift+R to rotate ghost piece
  if (e.key === 'r' || e.key === 'R') {
    const step = e.shiftKey ? -Math.PI / 2 : Math.PI / 2;
    ghostRotation += step;
    if (ghostTrack) {
      ghostTrack.rotation.y = ghostRotation;
    }
  }
  // Left/Right arrow keys or A/D to rotate ghost piece OR hovered placed track
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'a' || e.key === 'A' || e.key === 'd' || e.key === 'D') {
    const dir = (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') ? 1 : -1;
    // If ghost track is active (track not yet placed), rotate it
    if (ghostTrack) {
      ghostRotation += dir * Math.PI / 2;
      ghostTrack.rotation.y = ghostRotation;
      e.preventDefault();
    } else if (hoveredTrack) {
      const trackType = hoveredTrack.userData.type;
      if (trackType === 'straight' || trackType === 'curveLeft' || trackType === 'curveRight' || trackType === 'bridge' || trackType === 'slope' || trackType === 'bridgeCurveLeft' || trackType === 'bridgeCurveRight') {
        hoveredTrack.rotation.y += dir * Math.PI / 2;
        hoveredTrack.updateMatrixWorld(true);
        snapTrack(hoveredTrack);
        if (trainGroup) updateTrainPath();
        // Focus DOF on rotated track
        dofFocusOnTrack = true;
        dofTrackTarget = hoveredTrack;
        lastTrackInteraction = performance.now();
      }
      e.preventDefault();
    }
  }
  // Delete / Backspace to remove hovered or selected track
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (hoveredTrack) {
      const t = hoveredTrack;
      unhighlightTrack();
      removeTrack(t);
      updateCounter();
    } else if (selectedPlacedTrack) {
      const t = selectedPlacedTrack;
      deselectPlacedTrack();
      removeTrack(t);
      updateCounter();
    }
  }
  // Escape to deselect current tool → go back to selection
  if (e.key === 'Escape') {
    // If mid-drag, cancel the drag and restore controls
    if (isDragging && dragTrack) {
      // Snap the track back so it's not left floating
      if (!autoSnapTrack(dragTrack)) {
        snapTrack(dragTrack);
      }
      if (trainGroup) updateTrainPath();
      isDragging = false;
      dragTrack = null;
      controls.enabled = true;
      renderer.domElement.style.cursor = '';
    }
    activateSelectionTool();
  }
  // Ctrl+Z undo last placed track
  if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
    if (placedTracks.length > 0) {
      const last = placedTracks[placedTracks.length - 1];
      if (hoveredTrack === last) unhighlightTrack();
      removeTrack(last);
      updateCounter();
    }
  }
});

// ─── 3D Rotate Overlay Button ───
const rotateOverlayBtn = document.getElementById('rotateOverlayBtn');
const _overlayWorldPos = new THREE.Vector3();
const _overlayScreenPos = new THREE.Vector3();

function applySelectionHighlight(track) {
  track.traverse(child => {
    if ((child.isMesh || child.isInstancedMesh) && child.material) {
      if (!originalMaterials.has(child)) {
        originalMaterials.set(child, child.material);
      }
      const origMat = originalMaterials.get(child);
      let tintMat = _selectHighlightMatCache.get(origMat);
      if (!tintMat) {
        tintMat = origMat.clone();
        if (tintMat.color) tintMat.color.lerp(selectHighlightColor, 0.35);
        if (tintMat.emissive) {
          tintMat.emissive.set(0xcc3322);
          tintMat.emissiveIntensity = 0.12;
        }
        _selectHighlightMatCache.set(origMat, tintMat);
      }
      child.material = tintMat;
    }
  });
}

function removeSelectionHighlight(track) {
  track.traverse(child => {
    if ((child.isMesh || child.isInstancedMesh) && originalMaterials.has(child)) {
      child.material = originalMaterials.get(child);
    }
  });
}

function selectPlacedTrack(track) {
  if (selectedPlacedTrack === track) return;
  deselectPlacedTrack();
  if (!track || !rotatableTypes.has(track.userData.type)) return;
  selectedPlacedTrack = track;
  // Apply persistent reddish highlight
  applySelectionHighlight(track);
  rotateOverlayBtn.classList.add('visible');
}

function deselectPlacedTrack() {
  if (selectedPlacedTrack) {
    removeSelectionHighlight(selectedPlacedTrack);
  }
  selectedPlacedTrack = null;
  rotateOverlayBtn.classList.remove('visible');
}

function updateRotateOverlayPosition() {
  if (!selectedPlacedTrack) return;
  // Check if track still exists in scene
  if (!selectedPlacedTrack.parent) {
    deselectPlacedTrack();
    return;
  }
  selectedPlacedTrack.getWorldPosition(_overlayWorldPos);
  // Offset button above the track
  _overlayWorldPos.y += 0.8;
  _overlayScreenPos.copy(_overlayWorldPos).project(camera);
  // Check if behind camera
  if (_overlayScreenPos.z > 1) {
    rotateOverlayBtn.style.display = 'none';
    return;
  }
  const hw = window.innerWidth / 2;
  const hh = window.innerHeight / 2;
  const sx = _overlayScreenPos.x * hw + hw;
  const sy = -_overlayScreenPos.y * hh + hh;
  rotateOverlayBtn.style.left = sx + 'px';
  rotateOverlayBtn.style.top = sy + 'px';
}

rotateOverlayBtn.addEventListener('pointerenter', () => { isOverUI = true; });
rotateOverlayBtn.addEventListener('pointerleave', () => { isOverUI = false; });
rotateOverlayBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  if (!selectedPlacedTrack) return;
  selectedPlacedTrack.rotation.y += Math.PI / 2;
  selectedPlacedTrack.updateMatrixWorld(true);
  snapTrack(selectedPlacedTrack);
  if (trainGroup) updateTrainPath();
  updateBridgePillarVisibility();
  // Focus DOF
  dofFocusOnTrack = true;
  dofTrackTarget = selectedPlacedTrack;
  lastTrackInteraction = performance.now();
});

// ─── Night Mode Toggle ───
function applyNightMode(night) {
  if (night) {
    // Save current environment
    savedDayEnv = scene.environment;
    savedDayBg = scene.background;

    // Generate night env if not cached
    if (!nightEnvMap) {
      nightEnvMap = generateNightEnvMap();
    }
    scene.environment = nightEnvMap;
    scene.background = nightEnvMap;
    scene.environmentIntensity = 1.8; // boost the env map for ambient IBL

    // Dim the scene lights
    keyLight.intensity = nightLightSettings.keyIntensity;
    keyLight.color.set(nightLightSettings.keyColor);
    fillLight.intensity = nightLightSettings.fillIntensity;
    fillLight.color.set(nightLightSettings.fillSkyColor);
    fillLight.groundColor.set(nightLightSettings.fillGroundColor);
    rimLight.intensity = nightLightSettings.rimIntensity;
    rimLight.color.set(nightLightSettings.rimColor);
    renderer.toneMappingExposure = nightLightSettings.exposure;

    // Turn on night ambient
    nightAmbient.intensity = 0.55;
    nightAmbient.color.set(0x445577);

    // Darken ground slightly (tints the texture)
    groundMat.color.set('#5a5855');

    // Darken rain particles for night
    rainMat.color.set(0x445566);
    rainMat.opacity = 0.18;
    splashDropMat.color.set(0x556677);

    // Turn on train lights if train exists
    if (trainGroup) {
      enableTrainLights(true);
    }
    // Turn on scenery lamp posts
    sceneryLampMats.forEach(m => { m.emissive.set('#FFEEAA'); m.emissiveIntensity = 1.5; });
    sceneryLampLights.forEach(l => { l.intensity = 2.5; });
  } else {
    // Restore day environment
    if (savedDayEnv) {
      scene.environment = savedDayEnv;
      scene.background = savedDayBg;
    }
    scene.environmentIntensity = 1.0;

    // Restore day lights
    keyLight.intensity = dayLightSettings.keyIntensity;
    keyLight.color.set(dayLightSettings.keyColor);
    fillLight.intensity = dayLightSettings.fillIntensity;
    fillLight.color.set(dayLightSettings.fillSkyColor);
    fillLight.groundColor.set(dayLightSettings.fillGroundColor);
    rimLight.intensity = dayLightSettings.rimIntensity;
    rimLight.color.set(dayLightSettings.rimColor);
    renderer.toneMappingExposure = dayLightSettings.exposure;

    // Turn off night ambient
    nightAmbient.intensity = 0;

    // Restore ground
    groundMat.color.set('#9a9590');

    // Restore rain particles for day
    rainMat.color.set(0xaabbdd);
    rainMat.opacity = 0.25;
    splashDropMat.color.set(0xc8d8f0);

    // Turn off train lights
    if (trainGroup) {
      enableTrainLights(false);
    }
    // Turn off scenery lamp posts
    sceneryLampMats.forEach(m => { m.emissive.set('#000000'); m.emissiveIntensity = 0; });
    sceneryLampLights.forEach(l => { l.intensity = 0; });
  }
}

function enableTrainLights(on) {
  if (trainHeadlight) {
    trainHeadlight.intensity = on ? 60 : 0;
  }
  if (trainCabinLight) {
    trainCabinLight.intensity = on ? 4.0 : 0;
  }
  if (trainRearLight) {
    trainRearLight.intensity = on ? 3.5 : 0;
    if (trainRearLight._meshMat) {
      trainRearLight._meshMat.emissiveIntensity = on ? 5.0 : 0;
    }
  }
  if (trainHeadlightLensMat) {
    trainHeadlightLensMat.emissiveIntensity = on ? 15.0 : 0.6;
    trainHeadlightLensMat.emissive.set(on ? '#FFFFFF' : '#FFEE88');
    trainHeadlightLensMat.color.set(on ? '#FFFFFF' : '#FFFFCC');
  }
  if (trainFrontHubLight) {
    trainFrontHubLight.intensity = on ? 8.0 : 0;
  }
  if (trainFrontHubMat) {
    trainFrontHubMat.emissiveIntensity = on ? 6.0 : 0.4;
    trainFrontHubMat.emissive.set(on ? '#FFFFFF' : '#FFEE88');
    trainFrontHubMat.color.set(on ? '#FFFFDD' : '#C8A035');
  }
  // Window glow
  trainCabinWindowMats.forEach(m => {
    m.emissive.set(on ? '#FFAA33' : '#000000');
    m.emissiveIntensity = on ? 4.0 : 0;
  });
}

// ─── Build Toolbar UI ───
const toolbar = document.createElement('div');
toolbar.id = 'toolbar';
// Prevent toolbar pointer events from leaking to the canvas
toolbar.addEventListener('pointerdown', (e) => { e.stopPropagation(); isOverUI = true; });
toolbar.addEventListener('pointerup', (e) => { e.stopPropagation(); });
toolbar.addEventListener('pointerenter', () => { isOverUI = true; });
toolbar.addEventListener('pointerleave', () => { isOverUI = false; });

const trackTypes = [
  { id: 'straight', label: 'Straight', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><line x1="3" y1="12" x2="21" y2="12" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/></svg>' },
  { id: 'curveLeft', label: 'Curve', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><path d="M4 20 Q4 4 20 4" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round" fill="none"/></svg>' },
  { id: 'bridge', label: 'Bridge', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><line x1="3" y1="6" x2="21" y2="6" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="7" y1="6" x2="7" y2="20" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="17" y1="6" x2="17" y2="20" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/></svg>' },
  { id: 'slope', label: 'Slope', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><line x1="4" y1="20" x2="20" y2="4" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/></svg>' },
  { id: 'crossing', label: 'Cross', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><line x1="12" y1="3" x2="12" y2="21" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="3" y1="12" x2="21" y2="12" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/></svg>' },
  { id: 'bridgeCurveLeft', label: 'B.Curve', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><path d="M3 10 Q12 1 21 10" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round" fill="none"/><line x1="7" y1="8" x2="7" y2="20" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="17" y1="8" x2="17" y2="20" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/></svg>' },
];

// Selection tool button (arrow cursor)
const selectBtn = document.createElement('button');
selectBtn.className = 'tool-btn active'; // active by default
selectBtn.innerHTML = `<span class="tool-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><path d="M5 3L21 12L12 12L7 20Z" stroke="#B0B0B0" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" fill="none"/></svg></span><span class="tool-label">Select</span>`;
function activateSelectionTool() {
  selectionTool = true;
  selectedTrackType = null;
  rotateTool = false;
  document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  selectBtn.classList.add('active');
  removeGhost();
}
selectBtn.addEventListener('click', (e) => { e.stopPropagation(); activateSelectionTool(); });
toolbar.appendChild(selectBtn);

trackTypes.forEach(tt => {
  const btn = document.createElement('button');
  btn.className = 'tool-btn';
  btn.dataset.type = tt.id;
  btn.innerHTML = `<span class="tool-icon">${tt.icon}</span><span class="tool-label">${tt.label}</span>`;
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (selectedTrackType === tt.id) {
      selectedTrackType = null;
      btn.classList.remove('active');
      activateSelectionTool();
    } else {
      selectionTool = false;
      selectedTrackType = tt.id;
      document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Deactivate rotate tool
      rotateTool = false;
      rotateBtn.classList.remove('active');
    }
    removeGhost();
  });
  toolbar.appendChild(btn);
});

// Rotate tool button (not in toolbar, but variable kept for internal references)
const rotateBtn = document.createElement('button');
rotateBtn.className = 'tool-btn';
rotateBtn.innerHTML = `<span class="tool-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><path d="M17 2v5h-5" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M17 7A8 8 0 1 0 19.5 13" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/></svg></span><span class="tool-label">Rotate</span>`;

// Separator + Train button
const trainBtn = document.createElement('button');
trainBtn.className = 'tool-btn train-btn';
const _runIcon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><path d="M6 4L20 12L6 20V4Z" stroke="#B0B0B0" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/></svg>';
const _pauseIcon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><rect x="6" y="5" width="4" height="14" rx="1" stroke="#B0B0B0" stroke-width="2" stroke-linejoin="round"/><rect x="14" y="5" width="4" height="14" rx="1" stroke="#B0B0B0" stroke-width="2" stroke-linejoin="round"/></svg>';
const _stopIcon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><rect x="5" y="5" width="14" height="14" rx="2" stroke="#B0B0B0" stroke-width="2" stroke-linejoin="round"/></svg>';
trainBtn.innerHTML = `<span class="tool-icon">${_runIcon}</span><span class="tool-label">Run</span>`;

function updateTrainBtnUI() {
  if (trainRunning && !trainPaused) {
    // Train is running — show Pause button
    trainBtn.classList.add('active');
    trainBtn.classList.remove('paused');
    trainBtn.querySelector('.tool-label').textContent = 'Pause';
    trainBtn.querySelector('.tool-icon').innerHTML = _pauseIcon;
    stopTrainBtn.style.display = '';
  } else if (trainGroup && trainPaused) {
    // Train exists but is paused — show Resume button
    trainBtn.classList.remove('active');
    trainBtn.classList.add('paused');
    trainBtn.querySelector('.tool-label').textContent = 'Resume';
    trainBtn.querySelector('.tool-icon').innerHTML = _runIcon;
    stopTrainBtn.style.display = '';
  } else {
    // No train — show Run button
    trainBtn.classList.remove('active');
    trainBtn.classList.remove('paused');
    trainBtn.querySelector('.tool-label').textContent = 'Run';
    trainBtn.querySelector('.tool-icon').innerHTML = _runIcon;
    stopTrainBtn.style.display = 'none';
  }
}

// Stop train button (separate)
const stopTrainBtn = document.createElement('button');
stopTrainBtn.className = 'tool-btn train-btn stop-train-btn';
stopTrainBtn.innerHTML = `<span class="tool-icon">${_stopIcon}</span><span class="tool-label">Stop</span>`;
stopTrainBtn.style.display = 'none';

function removeTrainAndWagons() {
  if (trainGroup) {
    scene.remove(trainGroup);
    trainGroup.traverse(c => { if (c.geometry) c.geometry.dispose(); });
    trainGroup = null;
    trainHeadlight = null;
    trainHeadlightTarget = null;
    trainCabinLight = null;
    trainRearLight = null;
    trainFrontHubLight = null;
    trainFrontHubMat = null;
    trainHeadlightLensMat = null;
    trainCabinWindowMats = [];
    trainWheelMeshes = [];
    trainWheelRadii = [];
    Object.keys(trainMats).forEach(k => trainMats[k] = null);
  }
  for (const wg of wagonGroups) {
    if (wg) {
      scene.remove(wg);
      wg.traverse(c => { if (c.geometry) c.geometry.dispose(); });
    }
  }
  wagonGroups = [];
  wagonWheelMeshes = [];
  wagonCurrentYaws = [];
  wagonCurrentPitches = [];
  for (const cb of couplingBars) {
    if (cb.group) {
      scene.remove(cb.group);
      cb.group.traverse(c => { if (c.geometry) c.geometry.dispose(); });
    }
  }
  couplingBars = [];
  trainRunning = false;
  trainPaused = false;
  stopTrainBtn.style.display = 'none';
  updateTrainBtnUI();
}

stopTrainBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  removeTrainAndWagons();
});

trainBtn.addEventListener('click', (e) => {
  e.stopPropagation();

  if (trainRunning && !trainPaused) {
    // Currently running → pause
    trainPaused = true;
    trainRunning = false;
    updateTrainBtnUI();
    return;
  }

  if (trainGroup && trainPaused) {
    // Currently paused → resume
    trainPaused = false;
    trainRunning = true;
    trainStopped = false;
    updateTrainBtnUI();
    return;
  }

  // No train exists → create and run
  trainRunning = true;
  trainPaused = false;
  if (!trainGroup) {
    trainGroup = createTrain();
    scene.add(trainGroup);
    // Create wagons
    wagonGroups = [];
    wagonWheelMeshes = [];
    wagonCurrentYaws = [];
    wagonCurrentPitches = [];
    for (let wi = 0; wi < NUM_WAGONS; wi++) {
      const wagon = createWagon(wi);
      scene.add(wagon);
      wagonGroups.push(wagon);
      wagonWheelMeshes.push(wagon._wheels || []);
      wagonCurrentYaws.push(0);
      wagonCurrentPitches.push(0);
    }
    // Create coupling bars between train<->wagon and wagon<->wagon
    couplingBars = [];
    for (let ci = 0; ci < NUM_WAGONS; ci++) {
      const cb = createCouplingBar();
      cb.group.name = `coupling_bar_${ci}`;
      scene.add(cb.group);
      couplingBars.push(cb);
    }
    if (isNightMode) {
      enableTrainLights(true);
    }
  }
  trainStopped = false;
  updateTrainPath();
  trainT = 0;
  if (trainPathPoints.length > 0) {
    trainPrevPos.copy(trainPathPoints[0]);
    trainGroup.position.copy(trainPathPoints[0]);
    // Compute correct initial orientation from the path direction
    const initOri = getInitialOrientation(0, trainPathPoints);
    trainCurrentYaw = initOri.yaw;
    trainCurrentPitch = initOri.pitch;
    applyYawPitch(trainGroup, trainCurrentYaw, trainCurrentPitch);
    // Position and orient wagons correctly too
    for (let wIdx = 0; wIdx < wagonGroups.length; wIdx++) {
      const wg = wagonGroups[wIdx];
      if (!wg) continue;
      const wagonT = 0 - WAGON_GAP * (wIdx + 1);
      const { pos: wPos } = getPathPosition(wagonT, trainPathPoints);
      wg.position.copy(wPos);
      const wOri = getInitialOrientation(wagonT, trainPathPoints);
      wagonCurrentYaws[wIdx] = wOri.yaw;
      wagonCurrentPitches[wIdx] = wOri.pitch;
      applyYawPitch(wg, wOri.yaw, wOri.pitch);
      if (!_wagonPrevPositions[wIdx]) _wagonPrevPositions[wIdx] = wPos.clone();
      else _wagonPrevPositions[wIdx].copy(wPos);
    }
  } else {
    trainCurrentYaw = 0;
    trainCurrentPitch = 0;
  }
  stopTrainBtn.style.display = '';
  updateTrainBtnUI();
});

toolbar.appendChild(trainBtn);
toolbar.appendChild(stopTrainBtn);

// Train auto-start is deferred until after buildDefaultOval() below

// Night mode button
const nightBtn = document.createElement('button');
nightBtn.className = 'tool-btn train-btn';
nightBtn.innerHTML = `<span class="tool-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></span><span class="tool-label">Night</span>`;
nightBtn.addEventListener('click', () => {
  isNightMode = !isNightMode;
  nightBtn.classList.toggle('active', isNightMode);
  nightBtn.querySelector('.tool-label').textContent = isNightMode ? 'Day' : 'Night';
  nightBtn.querySelector('.tool-icon').innerHTML = isNightMode
    ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><circle cx="12" cy="12" r="5" stroke="#B0B0B0" stroke-width="2"/><line x1="12" y1="1" x2="12" y2="3" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="12" y1="21" x2="12" y2="23" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="1" y1="12" x2="3" y2="12" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="21" y1="12" x2="23" y2="12" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/></svg>'
    : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  applyNightMode(isNightMode);
});
nightBtn.addEventListener('pointerenter', () => { isOverUI = true; });
nightBtn.addEventListener('pointerleave', () => { isOverUI = false; });
toolbar.appendChild(nightBtn);

// Clear all button
const clearBtn = document.createElement('button');
clearBtn.className = 'tool-btn clear-btn';
clearBtn.innerHTML = `<span class="tool-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><line x1="6" y1="6" x2="18" y2="18" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/><line x1="18" y1="6" x2="6" y2="18" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round"/></svg></span><span class="tool-label">Clear</span>`;
clearBtn.addEventListener('click', () => {
  [...placedTracks].forEach(t => removeTrack(t));
  removeTrainAndWagons();
});
clearBtn.addEventListener('pointerenter', () => { isOverUI = true; });
clearBtn.addEventListener('pointerleave', () => { isOverUI = false; });
toolbar.appendChild(clearBtn);

root.appendChild(toolbar);

// Help text
const helpText = document.createElement('div');
helpText.id = 'helpText';
helpText.innerHTML = `Click to place · R to rotate ghost · Right-click or Del to remove · Drag to move · Rotate tool to spin placed tracks · Ctrl+Z undo`;
root.appendChild(helpText);

// Title
const titleEl = document.createElement('div');
titleEl.id = 'titleEl';
titleEl.innerHTML = `<span class="title-icon">🚂</span> Train Track Assembler`;
root.appendChild(titleEl);

// Track counter
const counterEl = document.createElement('div');
counterEl.id = 'counterEl';
function updateCounter() {
  counterEl.textContent = `${placedTracks.length} pieces`;
}
updateCounter();
root.appendChild(counterEl);

// Override placeTrack/removeTrack to update counter
const origPlace = placeTrack;
const origRemove = removeTrack;

// ─── Properties Panel ───
const propsToggle = document.createElement('button');
propsToggle.id = 'propsToggle';
propsToggle.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="display:block"><circle cx="12" cy="12" r="3" stroke="#B0B0B0" stroke-width="2"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="#B0B0B0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
propsToggle.addEventListener('pointerenter', () => { isOverUI = true; });
propsToggle.addEventListener('pointerleave', () => { isOverUI = false; });
root.appendChild(propsToggle);

const propsPanel = document.createElement('div');
propsPanel.id = 'propsPanel';
propsPanel.addEventListener('pointerenter', () => { isOverUI = true; });
propsPanel.addEventListener('pointerleave', () => { isOverUI = false; });
propsPanel.innerHTML = `
  <div class="props-scroll-inner">
  <div class="props-section" style="padding-top:8px">
    <div class="prop-row"><span class="prop-label">Rain</span><div class="prop-slider-wrap"><button class="prop-toggle on" id="propRain"></button></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Wood Material</div>
    <div class="prop-row"><span class="prop-label">Birch Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propBirchColor" value="#ffffff"></div></div>
    <div class="prop-row"><span class="prop-label">Pine Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propPineColor" value="#C4A46C"></div></div>
    <div class="prop-row"><span class="prop-label">Rail Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propRailColor" value="#a3a3a3"></div></div>
    <div class="prop-row"><span class="prop-label">Dark Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propDarkColor" value="#6B5340"></div></div>
    <div class="prop-row"><span class="prop-label">Roughness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propRoughness" min="0" max="100" value="32"><span class="prop-value" id="propRoughnessVal">0.32</span></div></div>
    <div class="prop-row"><span class="prop-label">Metalness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propWoodMetalness" min="0" max="100" value="0"><span class="prop-value" id="propWoodMetalnessVal">0.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Clearcoat</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propClearcoat" min="0" max="100" value="45"><span class="prop-value" id="propClearcoatVal">0.45</span></div></div>
    <div class="prop-row"><span class="prop-label">CC Roughness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propWoodCCRough" min="0" max="100" value="30"><span class="prop-value" id="propWoodCCRoughVal">0.30</span></div></div>
    <div class="prop-row"><span class="prop-label">Normal Scale</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propNormal" min="0" max="200" value="60"><span class="prop-value" id="propNormalVal">0.60</span></div></div>
    <div class="prop-row"><span class="prop-label">Bump Scale</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBump" min="0" max="100" value="12"><span class="prop-value" id="propBumpVal">0.12</span></div></div>
    <div class="prop-row"><span class="prop-label">Env Intensity</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propEnvInt" min="0" max="200" value="80"><span class="prop-value" id="propEnvIntVal">0.80</span></div></div>
    <div class="prop-row"><span class="prop-label">Sheen</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSheen" min="0" max="100" value="30"><span class="prop-value" id="propSheenVal">0.30</span></div></div>
    <div class="prop-row"><span class="prop-label">Sheen Roughness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSheenRough" min="0" max="100" value="50"><span class="prop-value" id="propSheenRoughVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Sheen Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propSheenColor" value="#e8d8c8"></div></div>
    <div class="prop-row"><span class="prop-label">IOR</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propWoodIOR" min="100" max="250" value="150"><span class="prop-value" id="propWoodIORVal">1.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Highlight Clamp</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propHighlightClamp" min="0" max="100" value="0"><span class="prop-value" id="propHighlightClampVal">0.00</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Train Material</div>
    <div class="prop-row"><span class="prop-label">Body Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propTrainBodyColor" value="#171B12"></div></div>
    <div class="prop-row"><span class="prop-label">Cabin Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propTrainCabinColor" value="#B8242C"></div></div>
    <div class="prop-row"><span class="prop-label">Roof Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propTrainRoofColor" value="#1A1A1A"></div></div>
    <div class="prop-row"><span class="prop-label">Stack Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propTrainStackColor" value="#C8A035"></div></div>
    <div class="prop-row"><span class="prop-label">Wheel Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propTrainWheelColor" value="#B8242C"></div></div>
    <div class="prop-row"><span class="prop-label">Trim Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propTrainTrimColor" value="#C8A035"></div></div>
    <div class="prop-row"><span class="prop-label">Roughness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainRoughness" min="0" max="100" value="32"><span class="prop-value" id="propTrainRoughnessVal">0.32</span></div></div>
    <div class="prop-row"><span class="prop-label">Clearcoat</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainClearcoat" min="0" max="100" value="45"><span class="prop-value" id="propTrainClearcoatVal">0.45</span></div></div>
    <div class="prop-row"><span class="prop-label">CC Roughness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainCCRough" min="0" max="100" value="30"><span class="prop-value" id="propTrainCCRoughVal">0.30</span></div></div>
    <div class="prop-row"><span class="prop-label">Normal Scale</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainNormal" min="0" max="200" value="60"><span class="prop-value" id="propTrainNormalVal">0.60</span></div></div>
    <div class="prop-row"><span class="prop-label">Bump Scale</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainBump" min="0" max="100" value="12"><span class="prop-value" id="propTrainBumpVal">0.12</span></div></div>
    <div class="prop-row"><span class="prop-label">Env Intensity</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainEnvInt" min="0" max="300" value="80"><span class="prop-value" id="propTrainEnvIntVal">0.80</span></div></div>
    <div class="prop-row"><span class="prop-label">Sheen</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainSheen" min="0" max="100" value="30"><span class="prop-value" id="propTrainSheenVal">0.30</span></div></div>
    <div class="prop-row"><span class="prop-label">Sheen Roughness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainSheenRough" min="0" max="100" value="50"><span class="prop-value" id="propTrainSheenRoughVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Sheen Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propTrainSheenColor" value="#e8d8c8"></div></div>
    <div class="prop-row"><span class="prop-label">IOR</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainIOR" min="100" max="250" value="150"><span class="prop-value" id="propTrainIORVal">1.50</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Train Reflections</div>
    <div class="prop-row"><span class="prop-label">Reflectivity</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainReflect" min="0" max="100" value="50"><span class="prop-value" id="propTrainReflectVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Metalness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainMetalness" min="0" max="100" value="0"><span class="prop-value" id="propTrainMetalnessVal">0.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Body Reflect</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainBodyReflect" min="0" max="100" value="50"><span class="prop-value" id="propTrainBodyReflectVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Cabin Reflect</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainCabinReflect" min="0" max="100" value="50"><span class="prop-value" id="propTrainCabinReflectVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Roof Reflect</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainRoofReflect" min="0" max="100" value="50"><span class="prop-value" id="propTrainRoofReflectVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Stack Reflect</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainStackReflect" min="0" max="100" value="50"><span class="prop-value" id="propTrainStackReflectVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Wheel Reflect</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainWheelReflect" min="0" max="100" value="50"><span class="prop-value" id="propTrainWheelReflectVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Trim Reflect</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propTrainTrimReflect" min="0" max="100" value="50"><span class="prop-value" id="propTrainTrimReflectVal">0.50</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Scenery Materials</div>
    <div class="prop-row"><span class="prop-label">Sign Post</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propSignPostColor" value="#888888"></div></div>
    <div class="prop-row"><span class="prop-label">Stop Sign</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propStopSignColor" value="#CC0000"></div></div>
    <div class="prop-row"><span class="prop-label">Green Sign</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propGreenSignColor" value="#1a6b2a"></div></div>
    <div class="prop-row"><span class="prop-label">Yellow Sign</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propYellowSignColor" value="#f0c030"></div></div>
    <div class="prop-row"><span class="prop-label">Tree Trunk</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propTreeTrunkColor" value="#5C3A1E"></div></div>
    <div class="prop-row"><span class="prop-label">Leaf Dark</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propLeafDarkColor" value="#2D5A1E"></div></div>
    <div class="prop-row"><span class="prop-label">Leaf Mid</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propLeafMidColor" value="#3A7A28"></div></div>
    <div class="prop-row"><span class="prop-label">Leaf Light</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propLeafLightColor" value="#4CAF50"></div></div>
    <div class="prop-row"><span class="prop-label">Bush Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propBushColor" value="#2E7D32"></div></div>
    <div class="prop-row"><span class="prop-label">Bench Wood</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propBenchWoodColor" value="#8B5E3C"></div></div>
    <div class="prop-row"><span class="prop-label">Lamp Post</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propLampPostColor" value="#1a1a1a"></div></div>
    <div class="prop-row"><span class="prop-label">Roughness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSceneryRoughness" min="0" max="100" value="55"><span class="prop-value" id="propSceneryRoughnessVal">0.55</span></div></div>
    <div class="prop-row"><span class="prop-label">Metalness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSceneryMetalness" min="0" max="100" value="10"><span class="prop-value" id="propSceneryMetalnessVal">0.10</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Lighting</div>
    <div class="prop-row"><span class="prop-label">Key Light</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propKeyInt" min="0" max="800" value="280"><span class="prop-value" id="propKeyIntVal">2.80</span></div></div>
    <div class="prop-row"><span class="prop-label">Fill Light</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propFillInt" min="0" max="300" value="90"><span class="prop-value" id="propFillIntVal">0.90</span></div></div>
    <div class="prop-row"><span class="prop-label">Rim Light</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propRimInt" min="0" max="800" value="300"><span class="prop-value" id="propRimIntVal">3.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Key Color</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propKeyColor" value="#ffeedd"></div></div>
    <div class="prop-row"><span class="prop-label">Shadows</span><div class="prop-slider-wrap"><button class="prop-toggle on" id="propShadows"></button></div></div>
    <div class="prop-row"><span class="prop-label">Shadow Softness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propShadowRadius" min="0" max="50" value="2"><span class="prop-value" id="propShadowRadiusVal">2</span></div></div>
    <div class="prop-row"><span class="prop-label">Blur Samples</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propShadowSamples" min="1" max="64" value="4"><span class="prop-value" id="propShadowSamplesVal">4</span></div></div>
    <div class="prop-row"><span class="prop-label">Shadow Bias</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propShadowBias" min="0" max="100" value="5"><span class="prop-value" id="propShadowBiasVal">-0.0005</span></div></div>
    <div class="prop-row"><span class="prop-label">Normal Bias</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propShadowNBias" min="0" max="200" value="4"><span class="prop-value" id="propShadowNBiasVal">0.04</span></div></div>
    <div class="prop-row"><span class="prop-label">Shadow Map</span><div class="prop-slider-wrap"><select class="prop-select" id="propShadowMap"><option value="256">256</option><option value="512" selected>512</option><option value="1024">1024</option><option value="2048">2048</option><option value="4096">4096</option></select></div></div>
    <div class="prop-row"><span class="prop-label">Shadow Opacity</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propShadowOpacity" min="0" max="100" value="100"><span class="prop-value" id="propShadowOpacityVal">1.00</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Ambient Occlusion</div>
    <div class="prop-row"><span class="prop-label">Enabled</span><div class="prop-slider-wrap"><button class="prop-toggle on" id="propAoEnabled"></button></div></div>
    <div class="prop-row"><span class="prop-label">Radius</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propAoRadius" min="1" max="200" value="10"><span class="prop-value" id="propAoRadiusVal">0.10</span></div></div>
    <div class="prop-row"><span class="prop-label">Intensity</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propAoScale" min="1" max="500" value="60"><span class="prop-value" id="propAoScaleVal">0.60</span></div></div>
    <div class="prop-row"><span class="prop-label">Thickness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propAoThickness" min="1" max="300" value="71"><span class="prop-value" id="propAoThicknessVal">0.71</span></div></div>
    <div class="prop-row"><span class="prop-label">Samples</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propAoSamples" min="4" max="64" value="4"><span class="prop-value" id="propAoSamplesVal">4</span></div></div>
    <div class="prop-row"><span class="prop-label">Dist. Exponent</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propAoDistExp" min="1" max="400" value="103"><span class="prop-value" id="propAoDistExpVal">1.03</span></div></div>
    <div class="prop-row"><span class="prop-label">Dist. Falloff</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propAoDistFall" min="1" max="100" value="25"><span class="prop-value" id="propAoDistFallVal">0.25</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Rendering</div>
    <div class="prop-row"><span class="prop-label">Tone Mapping</span><div class="prop-slider-wrap"><button class="prop-toggle on" id="propToneMappingEnabled"></button></div></div>
    <div class="prop-row"><span class="prop-label">Tone Map Type</span><div class="prop-slider-wrap"><select class="prop-select" id="propToneMap"><option value="ACES">ACES Filmic</option><option value="Cineon">Cineon</option><option value="Reinhard">Reinhard</option><option value="Linear">Linear</option><option value="AgX">AgX</option></select></div></div>
    <div class="prop-row"><span class="prop-label">Exposure</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propExposure" min="10" max="300" value="50"><span class="prop-value" id="propExposureVal">0.50</span></div></div>
    <div class="prop-row"><span class="prop-label">Contrast</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propContrast" min="0" max="100" value="0"><span class="prop-value" id="propContrastVal">0.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Render Scale</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propRenderScale" min="25" max="200" value="100"><span class="prop-value" id="propRenderScaleVal">1.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Pixel Ratio</span><div class="prop-slider-wrap"><select class="prop-select" id="propPixelRatio"><option value="0.5">0.5x</option><option value="0.75">0.75x</option><option value="1">1x</option><option value="1.5">1.5x</option><option value="2" selected>2x</option></select></div></div>
    <div class="prop-row"><span class="prop-label">Film Grain</span><div class="prop-slider-wrap"><button class="prop-toggle" id="propGrain"></button></div></div>
    <div class="prop-row"><span class="prop-label">Grain Amount</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propGrainAmt" min="0" max="40" value="7"><span class="prop-value" id="propGrainAmtVal">0.07</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Screen-Space Reflections</div>
    <div class="prop-row"><span class="prop-label">Enabled</span><div class="prop-slider-wrap"><button class="prop-toggle on" id="propSsrEnabled"></button></div></div>
    <div class="prop-row"><span class="prop-label">Strength</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSsrStrength" min="0" max="100" value="100"><span class="prop-value" id="propSsrStrengthVal">1.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Thickness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSsrThickness" min="1" max="100" value="21"><span class="prop-value" id="propSsrThicknessVal">0.21</span></div></div>
    <div class="prop-row"><span class="prop-label">Max Distance</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSsrMaxDist" min="10" max="500" value="160"><span class="prop-value" id="propSsrMaxDistVal">1.60</span></div></div>
    <div class="prop-row"><span class="prop-label">Fresnel Power</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSsrFresnel" min="10" max="500" value="157"><span class="prop-value" id="propSsrFresnelVal">1.57</span></div></div>
    <div class="prop-row"><span class="prop-label">Fade</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSsrFade" min="0" max="100" value="90"><span class="prop-value" id="propSsrFadeVal">0.90</span></div></div>
    <div class="prop-row"><span class="prop-label">March Steps</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSsrMarch" min="2" max="32" value="16" step="1"><span class="prop-value" id="propSsrMarchVal">16</span></div></div>
    <div class="prop-row"><span class="prop-label">Refine Steps</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propSsrRefine" min="0" max="8" value="1" step="1"><span class="prop-value" id="propSsrRefineVal">1</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Bloom</div>
    <div class="prop-row"><span class="prop-label">Enabled</span><div class="prop-slider-wrap"><button class="prop-toggle" id="propBloomEnabled"></button></div></div>
    <div class="prop-row"><span class="prop-label">Strength</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBloomStrength" min="0" max="200" value="35"><span class="prop-value" id="propBloomStrengthVal">0.35</span></div></div>
    <div class="prop-row"><span class="prop-label">Radius</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBloomRadius" min="0" max="100" value="40"><span class="prop-value" id="propBloomRadiusVal">0.40</span></div></div>
    <div class="prop-row"><span class="prop-label">Threshold</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBloomThreshold" min="0" max="200" value="85"><span class="prop-value" id="propBloomThresholdVal">0.85</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Depth of Field</div>
    <div class="prop-row"><span class="prop-label">Enabled</span><div class="prop-slider-wrap"><button class="prop-toggle" id="propDofEnabled"></button></div></div>
    <div class="prop-row"><span class="prop-label">Focus Distance</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propDofFocus" min="1" max="40" step="0.1" value="8.3"><span class="prop-value" id="propDofFocusVal">8.3</span></div></div>
    <div class="prop-row"><span class="prop-label">Aperture</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propDofAperture" min="5" max="500" value="500"><span class="prop-value" id="propDofApertureVal">5.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Bokeh Scale</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propDofBokeh" min="0" max="500" value="100"><span class="prop-value" id="propDofBokehVal">10.00</span></div></div>
  </div>

  <div class="props-section">
    <div class="props-section-title">Ground / Environment</div>
    <div class="prop-row"><span class="prop-label">BG Blur</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBgBlur" min="0" max="100" value="30"><span class="prop-value" id="propBgBlurVal">0.30</span></div></div>
    <div class="prop-row"><span class="prop-label">Env Intensity</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propEnvIntensity" min="0" max="300" value="100"><span class="prop-value" id="propEnvIntensityVal">1.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground Tint</span><div class="prop-slider-wrap"><input type="color" class="prop-color-input" id="propBaseplateColor" value="#9a9590"></div></div>
    <div class="prop-row"><span class="prop-label">Ground Roughness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpRoughness" min="0" max="100" value="22"><span class="prop-value" id="propBpRoughnessVal">0.22</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground Metalness</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpMetalness" min="0" max="100" value="0"><span class="prop-value" id="propBpMetalnessVal">0.00</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground Clearcoat</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpClearcoat" min="0" max="100" value="60"><span class="prop-value" id="propBpClearcoatVal">0.60</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground CC Rough</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpCCRough" min="0" max="100" value="53"><span class="prop-value" id="propBpCCRoughVal">0.53</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground Normal</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpNormal" min="0" max="300" value="137"><span class="prop-value" id="propBpNormalVal">1.37</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground AO Int</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpAO" min="0" max="200" value="127"><span class="prop-value" id="propBpAOVal">1.27</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground IOR</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpIOR" min="100" max="250" value="139"><span class="prop-value" id="propBpIORVal">1.39</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground Sheen</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpSheen" min="0" max="100" value="22"><span class="prop-value" id="propBpSheenVal">0.22</span></div></div>
    <div class="prop-row"><span class="prop-label">Ground Reflectivity</span><div class="prop-slider-wrap"><input type="range" class="prop-slider" id="propBpReflect" min="0" max="100" value="61"><span class="prop-value" id="propBpReflectVal">0.61</span></div></div>
    <div class="prop-row">
      <span class="prop-label">HDR Map</span>
      <div class="prop-slider-wrap">
        <div class="hdr-select-wrap" id="hdrSelectWrap">
          <button class="hdr-select-btn" id="hdrSelectBtn">
            <span class="hdr-dot" id="hdrDot" style="background:linear-gradient(135deg,#d4854a,#f0a560)"></span>
            <span id="hdrBtnLabel">Venice</span>
            <span class="hdr-arrow">▼</span>
          </button>
          <div class="hdr-dropdown" id="hdrDropdown"></div>
        </div>
      </div>
    </div>
  </div>
  </div>
`;
root.appendChild(propsPanel);

// Toggle panel
let panelOpen = false;
propsToggle.addEventListener('click', () => {
  panelOpen = !panelOpen;
  propsPanel.classList.toggle('open', panelOpen);
  propsToggle.classList.toggle('open', panelOpen);
});

propsPanel.addEventListener('pointerdown', (e) => e.stopPropagation());
propsPanel.addEventListener('pointermove', (e) => e.stopPropagation());

// ─── Property bindings ───
function sliderBind(id, valId, divisor, callback) {
  const slider = document.getElementById(id);
  const valEl = document.getElementById(valId);
  if (!slider) return;
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value) / divisor;
    if (valEl) valEl.textContent = v.toFixed(2);
    callback(v);
  });
}

const allWoodMats = [birchMat, pineMat, railMat, darkMat, pegMat];

// Helper to update a material color + sheen tint
function setMatColorWithSheen(mat, hex) {
  mat.color.set(hex);
  const c = new THREE.Color(hex);
  const h = { h: 0, s: 0, l: 0 };
  c.getHSL(h);
  mat.sheenColor.setHSL(h.h, h.s * 0.3, Math.min(1, h.l + 0.35));
}

// ── Wood Material Bindings ──
document.getElementById('propBirchColor').addEventListener('input', (e) => setMatColorWithSheen(birchMat, e.target.value));
document.getElementById('propPineColor').addEventListener('input', (e) => setMatColorWithSheen(pineMat, e.target.value));
document.getElementById('propRailColor').addEventListener('input', (e) => setMatColorWithSheen(railMat, e.target.value));
document.getElementById('propDarkColor').addEventListener('input', (e) => setMatColorWithSheen(darkMat, e.target.value));

sliderBind('propRoughness', 'propRoughnessVal', 100, (v) => {
  allWoodMats.forEach(m => m.roughness = v);
});
sliderBind('propWoodMetalness', 'propWoodMetalnessVal', 100, (v) => {
  allWoodMats.forEach(m => m.metalness = v);
});
sliderBind('propClearcoat', 'propClearcoatVal', 100, (v) => {
  allWoodMats.forEach(m => m.clearcoat = v);
});
sliderBind('propWoodCCRough', 'propWoodCCRoughVal', 100, (v) => {
  allWoodMats.forEach(m => m.clearcoatRoughness = v);
});
sliderBind('propNormal', 'propNormalVal', 100, (v) => {
  allWoodMats.forEach(m => m.normalScale.set(v, v));
});
sliderBind('propBump', 'propBumpVal', 100, (v) => {
  allWoodMats.forEach(m => m.bumpScale = v);
});
sliderBind('propEnvInt', 'propEnvIntVal', 100, (v) => {
  allWoodMats.forEach(m => m.envMapIntensity = v);
});
sliderBind('propSheen', 'propSheenVal', 100, (v) => {
  allWoodMats.forEach(m => m.sheen = v);
});
sliderBind('propSheenRough', 'propSheenRoughVal', 100, (v) => {
  allWoodMats.forEach(m => m.sheenRoughness = v);
});
document.getElementById('propSheenColor').addEventListener('input', (e) => {
  allWoodMats.forEach(m => m.sheenColor.set(e.target.value));
});
sliderBind('propWoodIOR', 'propWoodIORVal', 100, (v) => {
  allWoodMats.forEach(m => { if (m.ior !== undefined) m.ior = v; });
});
sliderBind('propHighlightClamp', 'propHighlightClampVal', 100, (v) => {
  highlightClampU.value = v;
});

// ── Train Material Bindings ──
function setTrainMatColor(matKey, hex) {
  const m = trainMats[matKey];
  if (m) {
    m.color.set(hex);
    const c = new THREE.Color(hex);
    const h = { h: 0, s: 0, l: 0 };
    c.getHSL(h);
    if (m.sheenColor) m.sheenColor.setHSL(h.h, h.s * 0.3, Math.min(1, h.l + 0.35));
  }
}

document.getElementById('propTrainBodyColor').addEventListener('input', (e) => setTrainMatColor('body', e.target.value));
document.getElementById('propTrainCabinColor').addEventListener('input', (e) => setTrainMatColor('cabin', e.target.value));
document.getElementById('propTrainRoofColor').addEventListener('input', (e) => setTrainMatColor('roof', e.target.value));
document.getElementById('propTrainStackColor').addEventListener('input', (e) => {
  setTrainMatColor('stack', e.target.value);
  setTrainMatColor('stackCap', e.target.value);
});
document.getElementById('propTrainWheelColor').addEventListener('input', (e) => setTrainMatColor('wheel', e.target.value));
document.getElementById('propTrainTrimColor').addEventListener('input', (e) => setTrainMatColor('trim', e.target.value));

sliderBind('propTrainRoughness', 'propTrainRoughnessVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => m.roughness = v);
});
sliderBind('propTrainClearcoat', 'propTrainClearcoatVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => m.clearcoat = v);
});
sliderBind('propTrainCCRough', 'propTrainCCRoughVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => m.clearcoatRoughness = v);
});
sliderBind('propTrainNormal', 'propTrainNormalVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => { if (m.normalScale) m.normalScale.set(v, v); });
});
sliderBind('propTrainBump', 'propTrainBumpVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => m.bumpScale = v);
});
sliderBind('propTrainEnvInt', 'propTrainEnvIntVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => m.envMapIntensity = v);
});
sliderBind('propTrainSheen', 'propTrainSheenVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => m.sheen = v);
});
sliderBind('propTrainSheenRough', 'propTrainSheenRoughVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => m.sheenRoughness = v);
});
document.getElementById('propTrainSheenColor').addEventListener('input', (e) => {
  getAllTrainWoodMats().forEach(m => m.sheenColor.set(e.target.value));
});
sliderBind('propTrainIOR', 'propTrainIORVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => { if (m.ior !== undefined) m.ior = v; });
});

// ── Train Reflections Bindings ──
sliderBind('propTrainReflect', 'propTrainReflectVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => { if (m.reflectivity !== undefined) m.reflectivity = v; });
  // Sync per-part sliders to global value
  const sv = Math.round(v * 100);
  const vt = v.toFixed(2);
  ['Body','Cabin','Roof','Stack','Wheel','Trim'].forEach(part => {
    const sl = document.getElementById('propTrain' + part + 'Reflect');
    const vl = document.getElementById('propTrain' + part + 'ReflectVal');
    if (sl) sl.value = sv;
    if (vl) vl.textContent = vt;
  });
});
sliderBind('propTrainMetalness', 'propTrainMetalnessVal', 100, (v) => {
  getAllTrainWoodMats().forEach(m => m.metalness = v);
});

// Per-part reflectivity overrides
sliderBind('propTrainBodyReflect', 'propTrainBodyReflectVal', 100, (v) => {
  if (trainMats.body) trainMats.body.reflectivity = v;
});
sliderBind('propTrainCabinReflect', 'propTrainCabinReflectVal', 100, (v) => {
  if (trainMats.cabin) trainMats.cabin.reflectivity = v;
});
sliderBind('propTrainRoofReflect', 'propTrainRoofReflectVal', 100, (v) => {
  if (trainMats.roof) trainMats.roof.reflectivity = v;
});
sliderBind('propTrainStackReflect', 'propTrainStackReflectVal', 100, (v) => {
  if (trainMats.stack) trainMats.stack.reflectivity = v;
  if (trainMats.stackCap) trainMats.stackCap.reflectivity = v;
});
sliderBind('propTrainWheelReflect', 'propTrainWheelReflectVal', 100, (v) => {
  if (trainMats.wheel) trainMats.wheel.reflectivity = v;
});
sliderBind('propTrainTrimReflect', 'propTrainTrimReflectVal', 100, (v) => {
  if (trainMats.trim) trainMats.trim.reflectivity = v;
});

// ── Scenery Material Bindings ──
const allSignMats = [stopSignMat, greenSignMat, yellowSignMat, brownSignMat];
const allLeafMats = [leafMatDark, leafMatMid, leafMatLight, bushMat1, bushMat2];

document.getElementById('propSignPostColor').addEventListener('input', (e) => signPostMat.color.set(e.target.value));
document.getElementById('propStopSignColor').addEventListener('input', (e) => stopSignMat.color.set(e.target.value));
document.getElementById('propGreenSignColor').addEventListener('input', (e) => greenSignMat.color.set(e.target.value));
document.getElementById('propYellowSignColor').addEventListener('input', (e) => yellowSignMat.color.set(e.target.value));
document.getElementById('propTreeTrunkColor').addEventListener('input', (e) => treeTrunkMat.color.set(e.target.value));
document.getElementById('propLeafDarkColor').addEventListener('input', (e) => leafMatDark.color.set(e.target.value));
document.getElementById('propLeafMidColor').addEventListener('input', (e) => leafMatMid.color.set(e.target.value));
document.getElementById('propLeafLightColor').addEventListener('input', (e) => leafMatLight.color.set(e.target.value));
document.getElementById('propBushColor').addEventListener('input', (e) => { bushMat1.color.set(e.target.value); bushMat2.color.set(new THREE.Color(e.target.value).offsetHSL(0, 0.02, 0.03)); });
document.getElementById('propBenchWoodColor').addEventListener('input', (e) => benchWoodMat.color.set(e.target.value));
document.getElementById('propLampPostColor').addEventListener('input', (e) => lampPostMat.color.set(e.target.value));

sliderBind('propSceneryRoughness', 'propSceneryRoughnessVal', 100, (v) => {
  [...allSignMats, signPostMat, treeTrunkMat, ...allLeafMats, benchWoodMat, benchMetalMat, lampPostMat].forEach(m => m.roughness = v);
});
sliderBind('propSceneryMetalness', 'propSceneryMetalnessVal', 100, (v) => {
  [...allSignMats, treeTrunkMat, ...allLeafMats, benchWoodMat].forEach(m => m.metalness = v);
});

// Lighting
sliderBind('propKeyInt', 'propKeyIntVal', 100, (v) => { keyLight.intensity = v; });
sliderBind('propFillInt', 'propFillIntVal', 100, (v) => { fillLight.intensity = v; });
sliderBind('propRimInt', 'propRimIntVal', 100, (v) => { rimLight.intensity = v; });

document.getElementById('propKeyColor').addEventListener('input', (e) => {
  keyLight.color.set(e.target.value);
});

document.getElementById('propShadows').addEventListener('click', function() {
  this.classList.toggle('on');
  const on = this.classList.contains('on');
  renderer.shadowMap.enabled = on;
  keyLight.castShadow = on;
});

sliderBind('propShadowRadius', 'propShadowRadiusVal', 1, (v) => {
  keyLight.shadow.radius = v;
  document.getElementById('propShadowRadiusVal').textContent = v.toFixed(0);
});

document.getElementById('propShadowSamples').addEventListener('input', function() {
  const v = parseInt(this.value);
  keyLight.shadow.blurSamples = v;
  document.getElementById('propShadowSamplesVal').textContent = v;
});

document.getElementById('propShadowBias').addEventListener('input', function() {
  const v = -parseFloat(this.value) / 10000;
  keyLight.shadow.bias = v;
  document.getElementById('propShadowBiasVal').textContent = v.toFixed(4);
});

document.getElementById('propShadowNBias').addEventListener('input', function() {
  const v = parseFloat(this.value) / 100;
  keyLight.shadow.normalBias = v;
  document.getElementById('propShadowNBiasVal').textContent = v.toFixed(2);
});

document.getElementById('propShadowMap').addEventListener('change', function() {
  const res = parseInt(this.value);
  keyLight.shadow.mapSize.set(res, res);
  if (keyLight.shadow.map) {
    keyLight.shadow.map.dispose();
    keyLight.shadow.map = null;
  }
});

sliderBind('propShadowOpacity', 'propShadowOpacityVal', 100, (v) => {
  keyLight.shadow.intensity = v;
});

// AO
document.getElementById('propAoEnabled').addEventListener('click', function() {
  this.classList.toggle('on');
  aoEnabledU.value = this.classList.contains('on') ? 1.0 : 0.0;
});

sliderBind('propAoRadius', 'propAoRadiusVal', 100, (v) => { aoPass.radius.value = v; });
sliderBind('propAoScale', 'propAoScaleVal', 100, (v) => { aoPass.scale.value = v; });
sliderBind('propAoThickness', 'propAoThicknessVal', 100, (v) => { aoPass.thickness.value = v; });
document.getElementById('propAoSamples').addEventListener('input', function() {
  const v = parseInt(this.value);
  aoPass.samples.value = v;
  document.getElementById('propAoSamplesVal').textContent = v;
});
sliderBind('propAoDistExp', 'propAoDistExpVal', 100, (v) => { aoPass.distanceExponent.value = v; });
sliderBind('propAoDistFall', 'propAoDistFallVal', 100, (v) => { aoPass.distanceFallOff.value = v; });

// Rendering
const toneMappingMap = {
  'ACES': THREE.ACESFilmicToneMapping,
  'Cineon': THREE.CineonToneMapping,
  'Reinhard': THREE.ReinhardToneMapping,
  'Linear': THREE.LinearToneMapping,
  'AgX': THREE.AgXToneMapping,
};
document.getElementById('propToneMappingEnabled').addEventListener('click', function() {
  this.classList.toggle('on');
  toneMappingEnabled = this.classList.contains('on');
  rebuildPostPipeline();
});

document.getElementById('propToneMap').addEventListener('change', (e) => {
  currentToneMapping = toneMappingMap[e.target.value] || THREE.ACESFilmicToneMapping;
  rebuildPostPipeline();
});

sliderBind('propExposure', 'propExposureVal', 100, (v) => { renderer.toneMappingExposure = v; });
sliderBind('propContrast', 'propContrastVal', 100, (v) => { contrastU.value = v; });

document.getElementById('propRenderScale').addEventListener('input', function() {
  renderScale = parseFloat(this.value) / 100;
  document.getElementById('propRenderScaleVal').textContent = renderScale.toFixed(2);
  const w = Math.floor(window.innerWidth * renderScale);
  const h = Math.floor(window.innerHeight * renderScale);
  renderer.setSize(w, h);
  renderer.domElement.style.width = window.innerWidth + 'px';
  renderer.domElement.style.height = window.innerHeight + 'px';
});

document.getElementById('propPixelRatio').addEventListener('change', function() {
  const v = parseFloat(this.value);
  renderer.setPixelRatio(v);
});

document.getElementById('propGrain').addEventListener('click', function() {
  this.classList.toggle('on');
  grainEnabledU.value = this.classList.contains('on') ? 1.0 : 0.0;
});

sliderBind('propGrainAmt', 'propGrainAmtVal', 100, (v) => { grainIntensityU.value = v; });

// Rain toggle
document.getElementById('propRain').addEventListener('click', function() {
  this.classList.toggle('on');
  rainEnabled = this.classList.contains('on');
  rainMesh.visible = rainEnabled;
  splashInstancedMesh.visible = rainEnabled;
});

// SSR
document.getElementById('propSsrEnabled').addEventListener('click', function() {
  this.classList.toggle('on');
  ssrEnabledU.value = this.classList.contains('on') ? 1.0 : 0.0;
});

sliderBind('propSsrStrength', 'propSsrStrengthVal', 100, (v) => { ssrStrengthU.value = v; });
sliderBind('propSsrThickness', 'propSsrThicknessVal', 100, (v) => { ssrThicknessU.value = v; });
sliderBind('propSsrMaxDist', 'propSsrMaxDistVal', 100, (v) => { ssrMaxDistU.value = v; });
sliderBind('propSsrFresnel', 'propSsrFresnelVal', 100, (v) => { ssrFresnelPowU.value = v; });
sliderBind('propSsrFade', 'propSsrFadeVal', 100, (v) => { ssrFadeU.value = v; });

document.getElementById('propSsrMarch').addEventListener('input', function() {
  const v = parseInt(this.value);
  ssrMarchSteps = v;
  document.getElementById('propSsrMarchVal').textContent = v;
  rebuildPostPipeline();
});

document.getElementById('propSsrRefine').addEventListener('input', function() {
  const v = parseInt(this.value);
  ssrRefineSteps = v;
  document.getElementById('propSsrRefineVal').textContent = v;
  rebuildPostPipeline();
});

// Bloom
document.getElementById('propBloomEnabled').addEventListener('click', function() {
  this.classList.toggle('on');
  bloomEnabled = this.classList.contains('on');
  rebuildPostPipeline();
});

sliderBind('propBloomStrength', 'propBloomStrengthVal', 100, (v) => { bloomStrengthU.value = v; });
sliderBind('propBloomRadius', 'propBloomRadiusVal', 100, (v) => { bloomRadiusU.value = v; });
sliderBind('propBloomThreshold', 'propBloomThresholdVal', 100, (v) => { bloomThresholdU.value = v; });

// DOF
document.getElementById('propDofEnabled').addEventListener('click', function() {
  this.classList.toggle('on');
  dofEnabled = this.classList.contains('on');
  rebuildPostPipeline();
});

document.getElementById('propDofFocus').addEventListener('input', function() {
  const v = parseFloat(this.value);
  focusDistU.value = v;
  document.getElementById('propDofFocusVal').textContent = v.toFixed(1);
});

sliderBind('propDofAperture', 'propDofApertureVal', 100, (v) => { focalLengthU.value = v; });
sliderBind('propDofBokeh', 'propDofBokehVal', 10, (v) => { bokehScaleU.value = v; });

// Environment
sliderBind('propBgBlur', 'propBgBlurVal', 100, (v) => { scene.backgroundBlurriness = v; });

document.getElementById('propBaseplateColor').addEventListener('input', (e) => {
  // Tint the concrete color map via material color multiply
  groundMat.color.set(e.target.value);
});

sliderBind('propBpRoughness', 'propBpRoughnessVal', 100, (v) => { groundMat.roughness = v; });
sliderBind('propBpMetalness', 'propBpMetalnessVal', 100, (v) => { groundMat.metalness = v; });
sliderBind('propBpClearcoat', 'propBpClearcoatVal', 100, (v) => { groundMat.clearcoat = v; });
sliderBind('propBpCCRough', 'propBpCCRoughVal', 100, (v) => { groundMat.clearcoatRoughness = v; });
sliderBind('propBpNormal', 'propBpNormalVal', 100, (v) => { groundMat.normalScale.set(v, v); });
sliderBind('propBpAO', 'propBpAOVal', 100, (v) => { groundMat.aoMapIntensity = v; });
sliderBind('propBpIOR', 'propBpIORVal', 100, (v) => { groundMat.ior = v; });
sliderBind('propBpSheen', 'propBpSheenVal', 100, (v) => { groundMat.sheen = v; });
sliderBind('propBpReflect', 'propBpReflectVal', 100, (v) => { groundMat.reflectivity = v; });
sliderBind('propEnvIntensity', 'propEnvIntensityVal', 100, (v) => { scene.environmentIntensity = v; });

// HDR dropdown
const hdrColors = {
  'Meadow': ['#6b8f4a', '#8faf6a'],
  'Venice': ['#d4854a', '#f0a560'],
  'Studio': ['#555555', '#888888'],
  'Forest': ['#4a7a3f', '#78b060'],
  'Urban': ['#6a7080', '#909aa8'],
  'Sunset': ['#d06030', '#f09050'],
  'Warehouse': ['#4a4a4a', '#787878'],
  'Night': ['#1a2040', '#304068'],
};

const hdrSelectBtn = document.getElementById('hdrSelectBtn');
const hdrDropdown = document.getElementById('hdrDropdown');
const hdrDot = document.getElementById('hdrDot');
const hdrBtnLabel = document.getElementById('hdrBtnLabel');
let hdrDropdownOpen = false;

HDR_ENVIRONMENTS.forEach((env, i) => {
  const colors = hdrColors[env.name] || ['#666', '#999'];
  const item = document.createElement('button');
  item.className = 'hdr-dropdown-item' + (i === 1 ? ' active' : '');
  item.innerHTML = `<span class="hdr-dot" style="background:linear-gradient(135deg,${colors[0]},${colors[1]})"></span><span>${env.name}</span><span class="hdr-check">✓</span>`;
  item.addEventListener('click', () => {
    loadHDR(i);
    hdrBtnLabel.textContent = env.name;
    hdrDot.style.background = `linear-gradient(135deg,${colors[0]},${colors[1]})`;
    hdrDropdown.querySelectorAll('.hdr-dropdown-item').forEach((b, j) => b.classList.toggle('active', j === i));
    hdrDropdownOpen = false;
    hdrDropdown.classList.remove('open');
    hdrSelectBtn.classList.remove('open');
  });
  hdrDropdown.appendChild(item);
});

hdrSelectBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  hdrDropdownOpen = !hdrDropdownOpen;
  hdrDropdown.classList.toggle('open', hdrDropdownOpen);
  hdrSelectBtn.classList.toggle('open', hdrDropdownOpen);
});

document.addEventListener('click', () => {
  if (hdrDropdownOpen) {
    hdrDropdownOpen = false;
    hdrDropdown.classList.remove('open');
    hdrSelectBtn.classList.remove('open');
  }
});

// ─── Place a default circular (oval) track ───

function buildDefaultOval() {
  // Chain: straight → curveRight × 2 → straight → curveRight × 2
  // This forms a closed oval loop

  const pieces = [
    'straight',
    'curveRight', 'curveRight',
    'straight',
    'curveRight', 'curveRight',
  ];

  // Place the first piece centered
  const first = createTrackByType(pieces[0]);
  first.position.set(0, 0, 0);
  first.name = `track_oval_0`;
  scene.add(first);
  placedTracks.push(first);
  first.updateMatrixWorld(true);

  // Chain subsequent pieces by aligning connector 0 of new piece to last connector of previous
  for (let i = 1; i < pieces.length; i++) {
    const p = createTrackByType(pieces[i]);
    p.name = `track_oval_${i}`;
    p.position.set(0, 0, 0);
    p.rotation.y = 0;
    scene.add(p);
    placedTracks.push(p);
    p.updateMatrixWorld(true);

    const prev = placedTracks[placedTracks.length - 2];
    const prevConns = getWorldConnectors(prev);
    // Exit = last connector of previous piece
    const exitConn = prevConns[prevConns.length - 1];

    // Entry = connector 0 of new piece
    const entryDirLocal = p.userData.connectors[0].dir.clone();

    // Target direction for entry connector: opposite of exit
    const targetDir = exitConn.dir.clone().negate();
    // Current entry dir in world
    const currentDir = entryDirLocal.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), p.rotation.y);
    const angle = Math.atan2(targetDir.x, targetDir.z) - Math.atan2(currentDir.x, currentDir.z);
    p.rotation.y += angle;
    p.updateMatrixWorld(true);

    // Position so connector 0 aligns with exit
    const myWorldConns = getWorldConnectors(p);
    const offset = exitConn.pos.clone().sub(myWorldConns[0].pos);
    p.position.add(offset);
    p.updateMatrixWorld(true);
  }

  // Center the entire oval on the baseplate origin
  const ovalBox = new THREE.Box3();
  for (const t of placedTracks) {
    ovalBox.expandByObject(t);
  }
  const ovalCenter = ovalBox.getCenter(new THREE.Vector3());
  ovalCenter.y = 0; // only shift in XZ, keep Y grounded
  for (const t of placedTracks) {
    t.position.sub(ovalCenter);
    t.updateMatrixWorld(true);
  }
}

buildDefaultOval();
updateCounter();

// Auto-start train AFTER oval is built so path is valid
stopTrainBtn.style.display = '';
trainPaused = false;
trainGroup = createTrain();
scene.add(trainGroup);
// Create wagons
wagonGroups = [];
wagonWheelMeshes = [];
wagonCurrentYaws = [];
wagonCurrentPitches = [];
for (let wi = 0; wi < NUM_WAGONS; wi++) {
  const wagon = createWagon(wi);
  scene.add(wagon);
  wagonGroups.push(wagon);
  wagonWheelMeshes.push(wagon._wheels || []);
  wagonCurrentYaws.push(0);
  wagonCurrentPitches.push(0);
}
// Create coupling bars between train<->wagon and wagon<->wagon
couplingBars = [];
for (let ci = 0; ci < NUM_WAGONS; ci++) {
  const cb = createCouplingBar();
  cb.group.name = `coupling_bar_${ci}`;
  scene.add(cb.group);
  couplingBars.push(cb);
}
updateTrainPath();
trainT = 0;
if (trainPathPoints.length > 0) {
  trainPrevPos.copy(trainPathPoints[0]);
  trainGroup.position.copy(trainPathPoints[0]);
  // Compute correct initial orientation from the path direction
  const initOri = getInitialOrientation(0, trainPathPoints);
  trainCurrentYaw = initOri.yaw;
  trainCurrentPitch = initOri.pitch;
  applyYawPitch(trainGroup, trainCurrentYaw, trainCurrentPitch);
  // Position and orient wagons correctly too
  for (let wIdx = 0; wIdx < wagonGroups.length; wIdx++) {
    const wg = wagonGroups[wIdx];
    if (!wg) continue;
    const wagonT = 0 - WAGON_GAP * (wIdx + 1);
    const { pos: wPos } = getPathPosition(wagonT, trainPathPoints);
    wg.position.copy(wPos);
    const wOri = getInitialOrientation(wagonT, trainPathPoints);
    wagonCurrentYaws[wIdx] = wOri.yaw;
    wagonCurrentPitches[wIdx] = wOri.pitch;
    applyYawPitch(wg, wOri.yaw, wOri.pitch);
    if (!_wagonPrevPositions[wIdx]) _wagonPrevPositions[wIdx] = wPos.clone();
    else _wagonPrevPositions[wIdx].copy(wPos);
  }
} else {
  trainCurrentYaw = 0;
  trainCurrentPitch = 0;
}
updateTrainBtnUI();

// ─── Scenery Props (stop signs, signs, mini-trees, bushes, lamp posts, bench) ───
const sceneryGroup = new THREE.Group();
sceneryGroup.name = 'sceneryGroup';
scene.add(sceneryGroup);

// Shared materials for scenery
const signPostMat = new THREE.MeshStandardMaterial({ color: '#888888', roughness: 0.4, metalness: 0.7 });
const stopSignMat = new THREE.MeshStandardMaterial({ color: '#CC0000', roughness: 0.5, metalness: 0.1 });
const stopTextMat = new THREE.MeshStandardMaterial({ color: '#FFFFFF', roughness: 0.5 });
const greenSignMat = new THREE.MeshStandardMaterial({ color: '#1a6b2a', roughness: 0.5, metalness: 0.1 });
const yellowSignMat = new THREE.MeshStandardMaterial({ color: '#f0c030', roughness: 0.4, metalness: 0.1 });
const brownSignMat = new THREE.MeshStandardMaterial({ color: '#8B6914', roughness: 0.55 });
const treeTrunkMat = new THREE.MeshStandardMaterial({ color: '#5C3A1E', roughness: 0.85 });
const leafMatDark = new THREE.MeshStandardMaterial({ color: '#2D5A1E', roughness: 0.8 });
const leafMatMid = new THREE.MeshStandardMaterial({ color: '#3A7A28', roughness: 0.78 });
const leafMatLight = new THREE.MeshStandardMaterial({ color: '#4CAF50', roughness: 0.75 });
const bushMat1 = new THREE.MeshStandardMaterial({ color: '#2E7D32', roughness: 0.85 });
const bushMat2 = new THREE.MeshStandardMaterial({ color: '#388E3C', roughness: 0.82 });
const benchWoodMat = new THREE.MeshStandardMaterial({ color: '#8B5E3C', roughness: 0.7 });
const benchMetalMat = new THREE.MeshStandardMaterial({ color: '#333333', roughness: 0.35, metalness: 0.8 });
const lampPostMat = new THREE.MeshStandardMaterial({ color: '#1a1a1a', roughness: 0.3, metalness: 0.85 });
const lampGlassMat = new THREE.MeshStandardMaterial({ color: '#FFFFDD', emissive: '#000000', emissiveIntensity: 0, roughness: 0.15, metalness: 0.05, transparent: true, opacity: 0.85 });

const sceneryLampMats = []; // track for night mode glow
const sceneryLampLights = []; // track point lights for night mode

// ── Shared scenery geometries (created once, reused) ──
const _stopPostGeo = new THREE.CylinderGeometry(0.03, 0.03, 1.2, 8);
const _stopSignShape = new THREE.Shape();
const _stopR = 0.22;
for (let i = 0; i < 8; i++) {
  const a = (Math.PI / 8) + (i * Math.PI / 4);
  if (i === 0) _stopSignShape.moveTo(Math.cos(a) * _stopR, Math.sin(a) * _stopR);
  else _stopSignShape.lineTo(Math.cos(a) * _stopR, Math.sin(a) * _stopR);
}
_stopSignShape.closePath();
const _stopSignGeo = new THREE.ExtrudeGeometry(_stopSignShape, { depth: 0.02, bevelEnabled: false });
const _stopBorderShape = new THREE.Shape();
const _stopRb = 0.2;
for (let i = 0; i < 8; i++) {
  const a = (Math.PI / 8) + (i * Math.PI / 4);
  if (i === 0) _stopBorderShape.moveTo(Math.cos(a) * _stopRb, Math.sin(a) * _stopRb);
  else _stopBorderShape.lineTo(Math.cos(a) * _stopRb, Math.sin(a) * _stopRb);
}
_stopBorderShape.closePath();
const _stopBorderGeo = new THREE.ExtrudeGeometry(_stopBorderShape, { depth: 0.025, bevelEnabled: false });
const _roadPostGeo = new THREE.CylinderGeometry(0.025, 0.025, 1.0, 8);
const _lampPostGeo = new THREE.CylinderGeometry(0.035, 0.045, 1.6, 8);
const _lampBaseGeo = new THREE.CylinderGeometry(0.1, 0.12, 0.08, 8);
const _lampArmGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.3, 6);
const _lampHousingGeo = new THREE.CylinderGeometry(0.08, 0.1, 0.12, 8);
const _lampGlassGeo = new THREE.SphereGeometry(0.065, 8, 6, 0, Math.PI * 2, 0, Math.PI / 2);
const _benchLegGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.3, 6);

// ── Helper: Create a stop sign ──
function createStopSign(x, z, rotY) {
  const group = new THREE.Group();
  group.name = `stopSign_${x}_${z}`;
  const post = new THREE.Mesh(_stopPostGeo, signPostMat);
  post.position.y = 0.6;
  post.castShadow = true;
  group.add(post);
  const sign = new THREE.Mesh(_stopSignGeo, stopSignMat);
  sign.position.set(0, 1.2, -0.01);
  sign.castShadow = true;
  group.add(sign);
  const inner = new THREE.Mesh(_stopBorderGeo, stopTextMat);
  inner.position.set(0, 1.2, -0.005);
  group.add(inner);
  group.position.set(x, 0, z);
  group.rotation.y = rotY || 0;
  sceneryGroup.add(group);
  return group;
}

// ── Helper: Create a rectangular road sign ──
function createRoadSign(x, z, rotY, mat, w, h) {
  const group = new THREE.Group();
  group.name = `roadSign_${x}_${z}`;
  const post = new THREE.Mesh(_roadPostGeo, signPostMat);
  post.position.y = 0.5;
  post.castShadow = true;
  group.add(post);
  const signGeo = new RoundedBoxGeometry(w || 0.5, h || 0.3, 0.02, 2, 0.01);
  const sign = new THREE.Mesh(signGeo, mat);
  sign.position.set(0, 1.05, 0);
  sign.castShadow = true;
  group.add(sign);
  group.position.set(x, 0, z);
  group.rotation.y = rotY || 0;
  sceneryGroup.add(group);
  return group;
}

// ── Helper: Create a mini tree ──
function createMiniTree(x, z, scale) {
  const s = scale || 1;
  const group = new THREE.Group();
  group.name = `miniTree_${x}_${z}`;
  // Trunk
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.05 * s, 0.07 * s, 0.6 * s, 8), treeTrunkMat);
  trunk.position.y = 0.3 * s;
  trunk.castShadow = true;
  group.add(trunk);
  // Foliage layers (3 stacked cones for a cartoony look)
  const leafMats = [leafMatDark, leafMatMid, leafMatLight];
  for (let i = 0; i < 3; i++) {
    const coneR = (0.35 - i * 0.08) * s;
    const coneH = (0.35 - i * 0.04) * s;
    const cone = new THREE.Mesh(new THREE.ConeGeometry(coneR, coneH, 8), leafMats[i]);
    cone.position.y = (0.6 + i * 0.22) * s;
    cone.castShadow = true;
    group.add(cone);
  }
  group.position.set(x, 0, z);
  sceneryGroup.add(group);
  return group;
}

// ── Helper: Create a round bush ──
function createBush(x, z, scale) {
  const s = scale || 1;
  const group = new THREE.Group();
  group.name = `bush_${x}_${z}`;
  const mat = Math.random() > 0.5 ? bushMat1 : bushMat2;
  // Main sphere
  const main = new THREE.Mesh(new THREE.SphereGeometry(0.2 * s, 8, 6), mat);
  main.position.y = 0.15 * s;
  main.castShadow = true;
  group.add(main);
  // A couple of smaller bumps
  for (let i = 0; i < 2; i++) {
    const bump = new THREE.Mesh(new THREE.SphereGeometry(0.12 * s, 6, 5), bushMat2);
    const angle = i * Math.PI + Math.random();
    bump.position.set(Math.cos(angle) * 0.12 * s, 0.1 * s, Math.sin(angle) * 0.12 * s);
    bump.castShadow = true;
    group.add(bump);
  }
  group.position.set(x, 0, z);
  sceneryGroup.add(group);
  return group;
}

// ── Helper: Create a park bench ──
function createBench(x, z, rotY) {
  const group = new THREE.Group();
  group.name = `bench_${x}_${z}`;
  // Seat
  const seat = new THREE.Mesh(new RoundedBoxGeometry(0.7, 0.04, 0.25, 2, 0.01), benchWoodMat);
  seat.position.y = 0.3;
  seat.castShadow = true;
  group.add(seat);
  // Backrest
  const back = new THREE.Mesh(new RoundedBoxGeometry(0.7, 0.25, 0.03, 2, 0.01), benchWoodMat);
  back.position.set(0, 0.48, -0.11);
  back.rotation.x = 0.12;
  back.castShadow = true;
  group.add(back);
  // Legs (4)
  const positions = [[-0.28, 0.15, -0.08], [0.28, 0.15, -0.08], [-0.28, 0.15, 0.08], [0.28, 0.15, 0.08]];
  positions.forEach((p, i) => {
    const leg = new THREE.Mesh(_benchLegGeo, benchMetalMat);
    leg.position.set(...p);
    leg.castShadow = true;
    leg.name = `benchLeg_${i}`;
    group.add(leg);
  });
  group.position.set(x, 0, z);
  group.rotation.y = rotY || 0;
  sceneryGroup.add(group);
  return group;
}

// ── Helper: Create a lamp post ──
function createLampPost(x, z) {
  const group = new THREE.Group();
  group.name = `lampPost_${x}_${z}`;
  const post = new THREE.Mesh(_lampPostGeo, lampPostMat);
  post.position.y = 0.8;
  post.castShadow = true;
  group.add(post);
  const base = new THREE.Mesh(_lampBaseGeo, lampPostMat);
  base.position.y = 0.04;
  group.add(base);
  const arm = new THREE.Mesh(_lampArmGeo, lampPostMat);
  arm.position.set(0.15, 1.55, 0);
  arm.rotation.z = Math.PI / 2;
  group.add(arm);
  const housing = new THREE.Mesh(_lampHousingGeo, lampPostMat);
  housing.position.set(0.3, 1.52, 0);
  group.add(housing);
  const glass = new THREE.Mesh(_lampGlassGeo, lampGlassMat.clone());
  glass.position.set(0.3, 1.46, 0);
  glass.rotation.x = Math.PI;
  group.add(glass);
  sceneryLampMats.push(glass.material);
  // Point light (off by default, turned on at night)
  const light = new THREE.PointLight(0xffeebb, 0, 3, 2);
  light.position.set(0.3, 1.44, 0);
  group.add(light);
  sceneryLampLights.push(light);
  group.position.set(x, 0, z);
  sceneryGroup.add(group);
  return group;
}

// ── Place scenery objects around the baseplate ──
// Stop signs
createStopSign(-5.5, -4, 0.3);
createStopSign(4.8, 5.2, Math.PI * 0.7);

// Road signs (green directional, yellow warning, brown info)
createRoadSign(-7, 2, 0.5, greenSignMat, 0.6, 0.25);
createRoadSign(6.5, -2.5, -0.4, yellowSignMat, 0.35, 0.35);
createRoadSign(2, 7, Math.PI, brownSignMat, 0.55, 0.3);
createRoadSign(-3, -7, 0.1, greenSignMat, 0.5, 0.25);

// Mini trees (scattered around the edges)
createMiniTree(-8, -6, 1.1);
createMiniTree(-7.5, 6.5, 0.9);
createMiniTree(7.5, 7, 1.2);
createMiniTree(8, -5, 0.8);
createMiniTree(-4, 8.5, 1.0);
createMiniTree(5, -8, 1.15);
createMiniTree(-9, 0, 0.85);
createMiniTree(9, 1.5, 1.05);
createMiniTree(0, 9, 0.95);
createMiniTree(-6, -8.5, 0.75);

// Bushes (fill gaps between trees)
createBush(-6.5, -5, 0.9);
createBush(-8.5, 5.5, 1.1);
createBush(6, 6, 0.8);
createBush(7, -3.5, 1.0);
createBush(-2, 8, 0.7);
createBush(3, -7.5, 0.85);
createBush(-8, 3, 0.95);
createBush(8.5, -1, 0.75);
createBush(1, -9, 0.9);
createBush(-5, 7.5, 1.05);
createBush(4, 8, 0.65);
createBush(-9, -3, 0.8);

// Park benches
createBench(-6, 4, Math.PI * 0.25);
createBench(6, -6, Math.PI * 0.75);

// Lamp posts (near benches / path areas)
createLampPost(-5.5, 5.5);
createLampPost(5.5, -5);
createLampPost(-8, -2);
createLampPost(7.5, 3.5);

// ─── Rain System ───
let rainEnabled = true;
const RAIN_COUNT = 8000;
const RAIN_AREA = 28;       // XZ spread
const RAIN_HEIGHT = 18;     // top of rain spawn
const RAIN_SPEED_MIN = 0.18;
const RAIN_SPEED_MAX = 0.32;
const RAIN_LENGTH = 0.35;

// Rain drop positions & velocities
const rainPositions = new Float32Array(RAIN_COUNT * 3);
const rainVelocities = new Float32Array(RAIN_COUNT);
for (let i = 0; i < RAIN_COUNT; i++) {
  rainPositions[i * 3]     = (Math.random() - 0.5) * RAIN_AREA;
  rainPositions[i * 3 + 1] = Math.random() * RAIN_HEIGHT;
  rainPositions[i * 3 + 2] = (Math.random() - 0.5) * RAIN_AREA;
  rainVelocities[i] = RAIN_SPEED_MIN + Math.random() * (RAIN_SPEED_MAX - RAIN_SPEED_MIN);
}

// Rain geometry – line segments (two vertices per drop)
const rainGeo = new THREE.BufferGeometry();
const rainVerts = new Float32Array(RAIN_COUNT * 6); // 2 verts × 3 components per drop
const rainIndices = [];
for (let i = 0; i < RAIN_COUNT; i++) {
  const x = rainPositions[i * 3];
  const y = rainPositions[i * 3 + 1];
  const z = rainPositions[i * 3 + 2];
  rainVerts[i * 6]     = x;
  rainVerts[i * 6 + 1] = y;
  rainVerts[i * 6 + 2] = z;
  rainVerts[i * 6 + 3] = x;
  rainVerts[i * 6 + 4] = y - RAIN_LENGTH;
  rainVerts[i * 6 + 5] = z;
  rainIndices.push(i * 2, i * 2 + 1);
}
rainGeo.setAttribute('position', new THREE.BufferAttribute(rainVerts, 3));
rainGeo.setIndex(rainIndices);

const rainMat = new THREE.LineBasicMaterial({
  color: 0xaabbdd,
  transparent: true,
  opacity: 0.25,
  depthWrite: false,
});
const rainMesh = new THREE.LineSegments(rainGeo, rainMat);
rainMesh.name = 'rainSystem';
rainMesh.frustumCulled = false;
rainMesh.visible = true;
scene.add(rainMesh);

// ─── Splash / Impact Particles (mesh-based for WebGPU compatibility) ───
const SPLASH_POOL = 600;
const splashPool = [];
const splashData = [];

// Create a shared tiny sphere geometry & material for splash droplets
const splashDropGeo = new THREE.SphereGeometry(0.018, 4, 3);
const splashDropMat = new THREE.MeshBasicMaterial({
  color: 0xc8d8f0,
  transparent: true,
  opacity: 0.85,
  depthWrite: false,
});

// Pre-allocate splash pool using InstancedMesh for performance
const splashDummy = new THREE.Object3D();
const splashInstancedMesh = new THREE.InstancedMesh(splashDropGeo, splashDropMat, SPLASH_POOL);
splashInstancedMesh.name = 'splashSystem';
splashInstancedMesh.frustumCulled = false;
// Hide all instances initially
for (let i = 0; i < SPLASH_POOL; i++) {
  splashDummy.position.set(0, -100, 0);
  splashDummy.scale.set(0, 0, 0);
  splashDummy.updateMatrix();
  splashInstancedMesh.setMatrixAt(i, splashDummy.matrix);
  splashData.push({
    alive: false,
    x: 0, y: -100, z: 0,
    vx: 0, vy: 0, vz: 0,
    life: 0, maxLife: 1,
    scale: 0,
  });
}
splashInstancedMesh.instanceMatrix.needsUpdate = true;
splashInstancedMesh.visible = true;
scene.add(splashInstancedMesh);

let splashHead = 0;

function spawnSplash(x, z) {
  const count = 4 + Math.floor(Math.random() * 5); // 4-8 particles per impact
  for (let n = 0; n < count; n++) {
    const i = splashHead;
    splashHead = (splashHead + 1) % SPLASH_POOL;
    const d = splashData[i];
    d.alive = true;
    d.x = x + (Math.random() - 0.5) * 0.06;
    d.y = 0.02;
    d.z = z + (Math.random() - 0.5) * 0.06;
    const angle = Math.random() * Math.PI * 2;
    const speed = 0.015 + Math.random() * 0.05;
    d.vx = Math.cos(angle) * speed;
    d.vy = 0.035 + Math.random() * 0.07;
    d.vz = Math.sin(angle) * speed;
    d.life = 0;
    d.maxLife = 10 + Math.random() * 16;
    d.scale = 0.6 + Math.random() * 1.2;
  }
}

function updateRain(dt) {
  const posAttr = rainGeo.getAttribute('position');
  const posArr = posAttr.array;

  for (let i = 0; i < RAIN_COUNT; i++) {
    const fall = rainVelocities[i] * dt;
    rainPositions[i * 3 + 1] -= fall;

    // Check if raindrop hit the ground (y <= 0)
    if (rainPositions[i * 3 + 1] <= 0) {
      const rx = rainPositions[i * 3];
      const rz = rainPositions[i * 3 + 2];
      // Spawn splash only for a fraction of drops (1 in 12) to keep pool manageable
      if (Math.abs(rx) <= 12 && Math.abs(rz) <= 12 && Math.random() < 0.08) {
        spawnSplash(rx, rz);
      }
      // Reset drop to top
      rainPositions[i * 3]     = (Math.random() - 0.5) * RAIN_AREA;
      rainPositions[i * 3 + 1] = RAIN_HEIGHT + Math.random() * 2;
      rainPositions[i * 3 + 2] = (Math.random() - 0.5) * RAIN_AREA;
      rainVelocities[i] = RAIN_SPEED_MIN + Math.random() * (RAIN_SPEED_MAX - RAIN_SPEED_MIN);
    }

    const x = rainPositions[i * 3];
    const y = rainPositions[i * 3 + 1];
    const z = rainPositions[i * 3 + 2];
    posArr[i * 6]     = x;
    posArr[i * 6 + 1] = y;
    posArr[i * 6 + 2] = z;
    posArr[i * 6 + 3] = x;
    posArr[i * 6 + 4] = y - RAIN_LENGTH;
    posArr[i * 6 + 5] = z;
  }
  posAttr.needsUpdate = true;

  // Update splash particles (instanced mesh)
  let splashDirty = false;
  for (let i = 0; i < SPLASH_POOL; i++) {
    const d = splashData[i];
    if (!d.alive) continue;
    splashDirty = true;
    d.life += dt;
    if (d.life >= d.maxLife) {
      d.alive = false;
      splashDummy.position.set(0, -100, 0);
      splashDummy.scale.set(0, 0, 0);
      splashDummy.updateMatrix();
      splashInstancedMesh.setMatrixAt(i, splashDummy.matrix);
    } else {
      const t = d.life / d.maxLife;
      d.x += d.vx * dt;
      d.y += d.vy * dt;
      d.z += d.vz * dt;
      // Gravity
      d.vy -= 0.004 * dt;
      // Clamp to ground
      if (d.y < 0.01) {
        d.y = 0.01;
        d.vy *= -0.12;
      }
      // Shrink + fade via scale
      const fade = (1.0 - t);
      const s = d.scale * fade;
      splashDummy.position.set(d.x, d.y, d.z);
      splashDummy.scale.set(s, s, s);
      splashDummy.updateMatrix();
      splashInstancedMesh.setMatrixAt(i, splashDummy.matrix);
    }
  }
  if (splashDirty) {
    splashInstancedMesh.instanceMatrix.needsUpdate = true;
  }
}

// ─── FPS Indicator ───
const fpsEl = document.createElement('div');
fpsEl.id = 'fpsIndicator';
fpsEl.textContent = '-- FPS';
root.appendChild(fpsEl);
let fpsFrames = 0;
let fpsLastUpdate = performance.now();

// ─── Animation ───
const _trainWorldPos = new THREE.Vector3(); // reusable — avoid per-frame allocation
let lastTime = performance.now();
let counterThrottle = 0;

function animate() {
  try {
  const now = performance.now();
  const dt = Math.min((now - lastTime) / 16.67, 3);
  lastTime = now;

  // FPS counter (update once per second to reduce DOM writes)
  fpsFrames++;
  if (now - fpsLastUpdate >= 1000) {
    const fps = Math.round(fpsFrames / ((now - fpsLastUpdate) / 1000));
    fpsEl.textContent = `${fps} FPS`;
    fpsFrames = 0;
    fpsLastUpdate = now;
  }

  controls.update();

  // Update rain & splashes
  if (rainEnabled) updateRain(dt);

  // Only update projection uniforms when camera actually moves (every frame is fine, it's cheap)
  projMatU.value.copy(camera.projectionMatrix);
  projInvMatU.value.copy(camera.projectionMatrixInverse);

  // Throttle DOM counter update
  if (++counterThrottle >= 30) {
    counterThrottle = 0;
    updateCounter();
  }

  if (trainRunning && trainGroup) {
    // Capture train position before move
    trainGroup.getWorldPosition(_prevTrainPos);

    moveTrainAlongPath(dt);

    trainGroup.getWorldPosition(_trainWorldPos);

    // Auto-resume follow after inactivity (but not while a track tool or selection is active)
    const trackToolActive = !!selectedTrackType || !!rotateTool;
    if (!cameraFollowActive && !selectedPlacedTrack && !trackToolActive && performance.now() - lastCameraInteraction > CAMERA_RESUME_DELAY) {
      cameraFollowActive = true;
      cameraFollowInit = false; // re-capture offset from current orbit position
    }

    if (cameraFollowActive && !selectedPlacedTrack && !trackToolActive) {
      // Camera follow logic
      if (!cameraFollowInit) {
        // First frame (or re-engage): store the offset from train to camera & controls target
        _camOffset.copy(camera.position).sub(_trainWorldPos);
        cameraFollowInit = true;
      }

      // Smoothly move camera and orbit target with the train
      const followSmooth = 1.0 - Math.pow(0.001, dt / 60);
      _targetCamPos.copy(_trainWorldPos).add(_camOffset);
      camera.position.lerp(_targetCamPos, followSmooth);
      controls.target.lerp(_trainWorldPos, followSmooth);

      // Guard against NaN from lerp (can freeze the whole renderer)
      if (isNaN(camera.position.x)) {
        camera.position.set(7, 2, 6.5);
        controls.target.set(0, 0, 0);
        cameraFollowInit = false;
      }
    }

  } else {
    cameraFollowInit = false;
  }

  // DOF focus: track interaction or train (runs regardless of trainRunning)
  if (dofFocusOnTrack && dofTrackTarget) {
    // Check if 3s elapsed since last track interaction
    if (now - lastTrackInteraction > DOF_TRACK_RESUME_DELAY) {
      dofFocusOnTrack = false;
      dofTrackTarget = null;
      dofZoomingWithSelection = false;
    } else if (dofTrackTarget.parent) {
      // Focus on the interacted track (only if still in scene)
      dofTrackTarget.getWorldPosition(_dofTrackWorldPos);
      const trackDist = camera.position.distanceTo(_dofTrackWorldPos);
      // Use faster lerp when actively zooming with selection for responsive DOF
      const lerpFactor = dofZoomingWithSelection && (now - lastSelectionZoomTime < 600) ? 0.35 : 0.15;
      const newFocus = focusDistU.value + (trackDist - focusDistU.value) * lerpFactor;
      if (isFinite(newFocus)) focusDistU.value = newFocus;
      // Clear zoom-with-selection flag after settling
      if (dofZoomingWithSelection && now - lastSelectionZoomTime > 600) {
        dofZoomingWithSelection = false;
      }
    } else {
      // Target was removed from scene — clear reference
      dofFocusOnTrack = false;
      dofTrackTarget = null;
      dofZoomingWithSelection = false;
    }
  }
  if (!dofFocusOnTrack && trainRunning && trainGroup) {
    // Auto-focus DOF on the train
    trainGroup.getWorldPosition(_trainWorldPos);
    const dist = camera.position.distanceTo(_trainWorldPos);
    const newFocus = focusDistU.value + (dist - focusDistU.value) * 0.1;
    if (isFinite(newFocus)) focusDistU.value = newFocus;
  }
  // Zoom-triggered DOF refocus on train (when nothing is selected and train exists but may not be running)
  if (dofZoomFocusTrain && !dofFocusOnTrack && trainGroup) {
    if (now - lastZoomTime > DOF_ZOOM_SETTLE_DELAY) {
      dofZoomFocusTrain = false;
    } else {
      trainGroup.getWorldPosition(_trainWorldPos);
      const dist = camera.position.distanceTo(_trainWorldPos);
      // Faster lerp during zoom for responsive feel
      const newFocus = focusDistU.value + (dist - focusDistU.value) * 0.18;
      if (isFinite(newFocus)) focusDistU.value = newFocus;
    }
  }

  // Lazily build the post-processing pipeline after a few clean frames
  if (!pipelineReady) {
    frameCount++;
    if (frameCount === PIPELINE_BUILD_FRAME) {
      rebuildPostPipeline();
      pipelineReady = true;
      // Now that rendering is smooth, start loading HDR in background
      setTimeout(() => loadHDR(1), 50);
    }
  }

  // Update the 3D rotate overlay button position each frame
  updateRotateOverlayPosition();

  if (pipelineReady) {
    postProcessing.render();
  } else {
    renderer.render(scene, camera);
  }

  } catch (e) {
    // Log but don't let a single frame error kill the animation loop
    console.warn('Animate frame error:', e);
  }
}

renderer.setAnimationLoop(animate);

let _resizeRAF = 0;
window.addEventListener('resize', () => {
  // Debounce resize to avoid repeated setSize calls during drag-resize
  cancelAnimationFrame(_resizeRAF);
  _resizeRAF = requestAnimationFrame(() => {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    // Skip resize if viewport collapsed (minimized tab, devtools, etc.) — prevents
    // WebGPU from receiving 0-dimension textures which can freeze the pipeline
    if (vw < 1 || vh < 1) return;
    camera.aspect = vw / vh;
    camera.updateProjectionMatrix();
    const w = Math.max(1, Math.floor(vw * renderScale));
    const h = Math.max(1, Math.floor(vh * renderScale));
    renderer.setSize(w, h);
    if (renderScale !== 1) {
      renderer.domElement.style.width = vw + 'px';
      renderer.domElement.style.height = vh + 'px';
    }
  });
});