"use client";

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { parseNPZ } from '@/lib/npz-parser';
import { infernoColormap } from '@/lib/colormaps';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Play, Pause } from 'lucide-react';

interface AnimationData {
  activity_timeline: Float32Array; // Shape: [n_sources, n_frames]
  timestamps: Float32Array; // Shape: [n_frames]
  source_positions: Float32Array; // Shape: [n_sources, 3] - source positions for activity
  mesh_vertices: Float32Array; // Shape: [n_mesh_vertices, 3] - full cortical mesh vertices
  triangles: Uint32Array; // Shape: [n_faces, 3] - references mesh_vertices
  fps: number;
}

interface AnimatedBrainVisualizationProps {
  animationFilePath: string; // Relative path like "results/edf_inference/.../animation_data.npz"
}

export function AnimatedBrainVisualization({ animationFilePath }: AnimatedBrainVisualizationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [data, setData] = useState<AnimationData | null>(null);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load NPZ file
  useEffect(() => {
    async function loadAnimation() {
      try {
        setLoading(true);
        setError(null);

        // The animationFilePath is like "results/edf_inference/.../animation_data.npz"
        // We need to fetch through the serve-result API
        // Remove 'results/' prefix if present since the API already adds it
        const pathWithoutResults = animationFilePath.startsWith('results/') 
          ? animationFilePath.substring('results/'.length) 
          : animationFilePath;
        
        const fetchUrl = `/api/serve-result/${pathWithoutResults}`;
        console.log('[AnimatedBrain] Fetching animation from:', fetchUrl);
        console.log('[AnimatedBrain] Original path:', animationFilePath);
        
        const response = await fetch(fetchUrl);
        if (!response.ok) {
          console.error('[AnimatedBrain] Fetch failed:', response.status, response.statusText);
          throw new Error(`Failed to fetch animation file: ${response.statusText}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        const npzData = await parseNPZ(arrayBuffer);

        // Extract and validate arrays (note: Python saves as 'activity', not 'activity_timeline')
        const activity = npzData['activity'] as Float32Array;
        const timestamps = npzData['timestamps'] as Float32Array;
        const source_positions = npzData['source_positions'] as Float32Array;
        const mesh_vertices = npzData['mesh_vertices'] as Float32Array;
        const triangles = npzData['triangles'] as Uint32Array;
        const fps_array = npzData['fps'] as Uint32Array;

        if (!activity || !timestamps || !source_positions || !triangles || !fps_array) {
          throw new Error('Missing required arrays in NPZ file');
        }

        // mesh_vertices may not exist in older NPZ files - warn but continue
        if (!mesh_vertices) {
          console.warn('mesh_vertices not found in NPZ - brain mesh overlay will not be rendered');
        }

        const fps = fps_array[0];

        // Note: activity is saved as (n_sources, n_frames) but we need (n_frames, n_sources)
        // We'll transpose it when accessing
        const n_sources = source_positions.length / 3;
        const n_mesh_vertices = mesh_vertices ? mesh_vertices.length / 3 : 0;
        const n_frames = timestamps.length;

        console.log('Animation data loaded:', {
          n_sources,
          n_mesh_vertices,
          n_frames,
          fps,
          activity_shape: `${n_sources} × ${n_frames}`,
          duration: timestamps[n_frames - 1].toFixed(2) + 's'
        });

        setData({
          activity_timeline: activity, // Store as-is, will transpose when accessing
          timestamps,
          source_positions,
          mesh_vertices: mesh_vertices || new Float32Array(0),
          triangles,
          fps,
        });
        setLoading(false);
      } catch (err) {
        console.error('Error loading animation:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    }

    loadAnimation();
  }, [animationFilePath]);

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current || !data) return;

    console.log('Initializing Three.js scene...', {
      containerWidth: containerRef.current.clientWidth,
      containerHeight: containerRef.current.clientHeight,
      n_vertices: data.source_positions.length / 3,
      n_triangles: data.triangles.length / 3,
    });

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      2000
    );
    camera.position.set(0, 0, 300); // Move camera further back
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    console.log('Renderer created and canvas appended:', {
      canvasWidth: renderer.domElement.width,
      canvasHeight: renderer.domElement.height,
      canvasStyle: renderer.domElement.style.cssText,
    });

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Create source points visualization (heatmap)
    const n_vertices = data.source_positions.length / 3;

    console.log('Creating source point cloud:', { n_vertices });

    // Create source points geometry
    const sourceGeometry = new THREE.BufferGeometry();
    sourceGeometry.setAttribute('position', new THREE.BufferAttribute(data.source_positions, 3));

    // Colors (will be updated per frame) - initialize to white for visibility
    const colors = new Float32Array(n_vertices * 3);
    for (let i = 0; i < n_vertices; i++) {
      colors[i * 3] = 1.0;     // R
      colors[i * 3 + 1] = 1.0; // G
      colors[i * 3 + 2] = 1.0; // B
    }
    sourceGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Use PointsMaterial for source visualization with larger points
    const sourceMaterial = new THREE.PointsMaterial({
      size: 5.0,
      vertexColors: true,
      sizeAttenuation: false, // Keep constant size
    });

    const sourcePoints = new THREE.Points(sourceGeometry, sourceMaterial);
    scene.add(sourcePoints);
    meshRef.current = sourcePoints as any; // Store for color updates

    // Center the geometry
    sourceGeometry.computeBoundingBox();
    const boundingBox = sourceGeometry.boundingBox!;
    const center = new THREE.Vector3();
    boundingBox.getCenter(center);
    sourceGeometry.translate(-center.x, -center.y, -center.z);

    console.log('Source points created:', {
      vertices: n_vertices,
      boundingBox: {
        min: boundingBox.min,
        max: boundingBox.max,
        center: center,
      },
    });

    // === BRAIN MESH OVERLAY ===
    // Create a semi-transparent brain surface mesh using the full mesh vertices and triangles
    let brainGeometry: THREE.BufferGeometry | null = null;
    let brainMaterial: THREE.MeshStandardMaterial | null = null;
    
    if (data.mesh_vertices.length > 0) {
      brainGeometry = new THREE.BufferGeometry();
      const brainPositions = data.mesh_vertices.slice(); // Clone mesh vertices (full cortical surface)
      brainGeometry.setAttribute('position', new THREE.BufferAttribute(brainPositions, 3));
      brainGeometry.setIndex(new THREE.BufferAttribute(data.triangles, 1));
      brainGeometry.computeVertexNormals();

      // Apply the same centering transformation
      brainGeometry.translate(-center.x, -center.y, -center.z);

      // Semi-transparent material for brain surface
      brainMaterial = new THREE.MeshStandardMaterial({
        color: 0x888888,          // Gray brain surface
        transparent: true,
        opacity: 0.3,             // Semi-transparent (adjust 0.1–0.5 as needed)
        side: THREE.DoubleSide,   // Render both sides of faces
        depthWrite: false,        // Prevent z-fighting with source points
      });

      const brainMesh = new THREE.Mesh(brainGeometry, brainMaterial);
      scene.add(brainMesh);

      console.log('Brain mesh overlay created:', {
        meshVertices: data.mesh_vertices.length / 3,
        triangles: data.triangles.length / 3,
        opacity: brainMaterial.opacity,
      });
    } else {
      console.warn('No mesh_vertices available - brain mesh overlay not created');
    }

    // Animation loop
    function animate() {
      animationFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // Handle window resize
    function handleResize() {
      if (!containerRef.current || !camera || !renderer) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    }
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
      }
      renderer.dispose();
      sourceGeometry.dispose();
      sourceMaterial.dispose();
      if (brainGeometry) brainGeometry.dispose();
      if (brainMaterial) brainMaterial.dispose();
      if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
    };
  }, [data]);

  // Update colors when frame changes
  useEffect(() => {
    if (!data || !meshRef.current) return;

    const n_vertices = data.source_positions.length / 3;
    const n_frames = data.timestamps.length;

    // Clamp frame to valid range
    const frame = Math.max(0, Math.min(currentFrame, n_frames - 1));

    // Get activity for this frame
    // Note: data is stored as (n_sources, n_frames), so we need to extract column `frame`
    const frameActivity = new Float32Array(n_vertices);
    for (let i = 0; i < n_vertices; i++) {
      frameActivity[i] = data.activity_timeline[i * n_frames + frame];
    }

    // Normalize activity to [0, 1]
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < frameActivity.length; i++) {
      if (frameActivity[i] < min) min = frameActivity[i];
      if (frameActivity[i] > max) max = frameActivity[i];
    }
    const range = max - min || 1; // Avoid division by zero

    // Update vertex colors
    const colorAttribute = meshRef.current.geometry.getAttribute('color') as THREE.BufferAttribute;
    for (let i = 0; i < n_vertices; i++) {
      const normalizedValue = (frameActivity[i] - min) / range;
      const rgb = infernoColormap(normalizedValue);
      colorAttribute.setXYZ(i, rgb[0], rgb[1], rgb[2]);
    }
    colorAttribute.needsUpdate = true;
  }, [data, currentFrame]);

  // Playback control
  useEffect(() => {
    if (!isPlaying || !data) {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const n_frames = data.timestamps.length;
    const frameInterval = 1000 / data.fps;

    intervalRef.current = setInterval(() => {
      setCurrentFrame((prevFrame) => {
        const nextFrame = prevFrame + 1;
        if (nextFrame >= n_frames) {
          setIsPlaying(false); // Stop at end
          return n_frames - 1;
        }
        return nextFrame;
      });
    }, frameInterval);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, data]);

  // Loading state
  if (loading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900">
        <p className="text-white">Loading animation...</p>
      </div>
    );
  }

  // Error state
  if (error || !data) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-900">
        <p className="text-red-500">Error: {error || 'No data loaded'}</p>
      </div>
    );
  }

  const n_frames = data.timestamps.length;
  const currentTimestamp = data.timestamps[currentFrame];

  return (
    <div className="w-full h-full flex flex-col bg-gray-900">
      {/* 3D Viewport */}
      <div ref={containerRef} className="flex-1 w-full min-h-[400px]" />

      {/* Playback Controls */}
      <div className="p-4 bg-gray-800 border-t border-gray-700 space-y-3">
        {/* Play/Pause + Timestamp */}
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setIsPlaying(!isPlaying)}
            className="shrink-0"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>

          <div className="text-sm text-white">
            Frame {currentFrame + 1} / {n_frames}
          </div>

          <div className="text-sm text-gray-400">
            {currentTimestamp.toFixed(2)}s
          </div>
        </div>

        {/* Timeline Slider */}
        <Slider
          value={[currentFrame]}
          onValueChange={(value) => {
            setCurrentFrame(value[0]);
            setIsPlaying(false); // Pause when scrubbing
          }}
          min={0}
          max={n_frames - 1}
          step={1}
          className="w-full"
        />
      </div>
    </div>
  );
}
