/**
 * SmartVisionOC - Multi-Layer Canvas Inspection Workstation
 * Handles interactive blending of Raw, CLAHE, CLIP Heatmap, SAM 2 Mask, and Depth Topography.
 */

export class CanvasViewer {
  constructor(canvasElement, coordDisplayElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.coordDisplay = coordDisplayElement;
    
    this.viewMode = 'composite'; // composite, raw, clahe, heatmap, sam, depth, quad
    this.alpha = 0.65;
    
    // Cached Image Objects
    this.images = {
      raw: null,
      clahe: null,
      heatmap: null,
      mask: null,
      depth: null
    };
    
    this.currentData = null;
    this.initEvents();
  }

  initEvents() {
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.canvas.width / rect.width;
      const scaleY = this.canvas.height / rect.height;
      const x = Math.round((e.clientX - rect.left) * scaleX);
      const y = Math.round((e.clientY - rect.top) * scaleY);
      
      if (this.coordDisplay) {
        this.coordDisplay.textContent = `X: ${x} px | Y: ${y} px`;
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      if (this.coordDisplay) {
        this.coordDisplay.textContent = `X: --- | Y: ---`;
      }
    });
  }

  async reset(previewImageSrc = null) {
    this.currentData = null;
    this.images = {
      raw: null,
      clahe: null,
      heatmap: null,
      mask: null,
      depth: null
    };

    if (previewImageSrc) {
      const img = new Image();
      img.onload = () => {
        this.images.raw = img;
        this.render();
      };
      img.src = previewImageSrc;
    } else {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  async loadInspectionData(pipelineData) {
    this.currentData = pipelineData;
    
    const loadImage = (src) => {
      return new Promise((resolve) => {
        if (!src) return resolve(null);
        const img = new Image();
        if (src.startsWith('http://') || src.startsWith('https://')) {
          img.crossOrigin = 'anonymous';
        }
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = src;
      });
    };

    const [raw, clahe, heatmap, mask, depth] = await Promise.all([
      loadImage(pipelineData.section_1_acquisition.image_base64),
      loadImage(pipelineData.section_2_preprocessing.enhanced_image_base64),
      loadImage(pipelineData.section_3_defect_identification.heatmap_base64),
      loadImage(pipelineData.section_4_segmentation.mask_base64),
      loadImage(pipelineData.section_5_spatial_analysis.depth_map_base64)
    ]);

    this.images = { raw, clahe, heatmap, mask, depth };
    this.render();
  }

  setViewMode(mode) {
    this.viewMode = mode;
    this.render();
  }

  setAlpha(alphaValue) {
    this.alpha = parseFloat(alphaValue);
    this.render();
  }

  render() {
    if (!this.images.raw) return;

    const w = this.images.raw.width || 1024;
    const h = this.images.raw.height || 1024;
    this.canvas.width = w;
    this.canvas.height = h;

    this.ctx.clearRect(0, 0, w, h);

    if (this.viewMode === 'quad') {
      this.renderQuadView(w, h);
      return;
    }

    const baseImg = this.images.clahe || this.images.raw;

    if (this.viewMode === 'raw') {
      this.ctx.drawImage(this.images.raw, 0, 0, w, h);
    } else if (this.viewMode === 'clahe') {
      this.ctx.drawImage(baseImg, 0, 0, w, h);
    } else if (this.viewMode === 'heatmap') {
      // Underlay base image so the defect heat signature is visually grounded
      this.ctx.drawImage(baseImg, 0, 0, w, h);
      if (this.images.heatmap) {
        this.ctx.save();
        this.ctx.globalAlpha = this.alpha;
        this.ctx.drawImage(this.images.heatmap, 0, 0, w, h);
        this.ctx.restore();
      }
    } else if (this.viewMode === 'sam') {
      this.ctx.drawImage(baseImg, 0, 0, w, h);
      if (this.images.mask) {
        this.ctx.drawImage(this.images.mask, 0, 0, w, h);
      }
    } else if (this.viewMode === 'depth') {
      if (this.images.depth) {
        this.ctx.drawImage(this.images.depth, 0, 0, w, h);
      } else {
        this.ctx.drawImage(baseImg, 0, 0, w, h);
      }
    } else {
      // COMPOSITE VIEW
      // 1. Draw base enhanced image
      this.ctx.drawImage(baseImg, 0, 0, w, h);

      // 2. Blend Heatmap with Alpha
      if (this.images.heatmap) {
        this.ctx.save();
        this.ctx.globalAlpha = this.alpha;
        this.ctx.drawImage(this.images.heatmap, 0, 0, w, h);
        this.ctx.restore();
      }

      // 3. Draw SAM 2 Segmented Mask Overlay
      if (this.images.mask) {
        this.ctx.save();
        this.ctx.globalAlpha = Math.min(1.0, this.alpha + 0.25);
        this.ctx.drawImage(this.images.mask, 0, 0, w, h);
        this.ctx.restore();
      }

      // 4. Draw Bounding Box and Centroid Reticle whenever a defect is localized
      if (this.currentData && this.currentData.section_4_segmentation) {
        const seg = this.currentData.section_4_segmentation;
        const defectName = this.currentData.section_3_defect_identification?.predicted_defect;
        const bbox = seg.bounding_box;
        const centroid = seg.centroid;

        if (defectName && defectName !== "Normal" && bbox && bbox[2] > 0 && bbox[3] > 0) {
          const bx = bbox[0], by = bbox[1], bw = bbox[2] - bbox[0], bh = bbox[3] - bbox[1];
          
          this.ctx.save();
          this.ctx.strokeStyle = '#ef4444';
          this.ctx.lineWidth = 2;
          this.ctx.setLineDash([6, 4]);
          this.ctx.strokeRect(bx, by, bw, bh);

          // Bounding Box Tag
          this.ctx.fillStyle = '#ef4444';
          this.ctx.fillRect(bx, Math.max(0, by - 22), 160, 20);
          this.ctx.fillStyle = '#ffffff';
          this.ctx.font = 'bold 11px JetBrains Mono, monospace';
          this.ctx.fillText(`DEFECT: ${defectName}`, bx + 6, Math.max(14, by - 8));

          // Reticle Centroid
          if (centroid && (centroid[0] > 0 || centroid[1] > 0)) {
            const cx = centroid[0], cy = centroid[1];
            this.ctx.strokeStyle = '#38bdf8';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([]);
            this.ctx.beginPath();
            this.ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
            this.ctx.moveTo(cx - 12, cy); this.ctx.lineTo(cx + 12, cy);
            this.ctx.moveTo(cx, cy - 12); this.ctx.lineTo(cx, cy + 12);
            this.ctx.stroke();
          }
          this.ctx.restore();
        }
      }
    }
  }

  renderQuadView(w, h) {
    const halfW = w / 2;
    const halfH = h / 2;
    const baseImg = this.images.clahe || this.images.raw;

    // Top-Left: Raw
    if (this.images.raw) {
      this.ctx.drawImage(this.images.raw, 0, 0, halfW, halfH);
    }
    this.drawQuadLabel(10, 24, "1. RAW CAPTURE");

    // Top-Right: CLAHE
    if (baseImg) {
      this.ctx.drawImage(baseImg, halfW, 0, halfW, halfH);
    }
    this.drawQuadLabel(halfW + 10, 24, "2. CLAHE ENHANCED");

    // Bottom-Left: Heatmap (draw base image first, then heatmap overlay)
    if (baseImg) {
      this.ctx.drawImage(baseImg, 0, halfH, halfW, halfH);
    }
    if (this.images.heatmap) {
      this.ctx.save();
      this.ctx.globalAlpha = 0.75;
      this.ctx.drawImage(this.images.heatmap, 0, halfH, halfW, halfH);
      this.ctx.restore();
    }
    this.drawQuadLabel(10, halfH + 24, "3. CLIP HEATMAP");

    // Bottom-Right: Depth Topography
    if (this.images.depth) {
      this.ctx.drawImage(this.images.depth, halfW, halfH, halfW, halfH);
    } else if (baseImg) {
      this.ctx.drawImage(baseImg, halfW, halfH, halfW, halfH);
    }
    this.drawQuadLabel(halfW + 10, halfH + 24, "4. DEPTH PROFILE");

    // Grid divider lines
    this.ctx.strokeStyle = '#3b82f6';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(halfW, 0); this.ctx.lineTo(halfW, h);
    this.ctx.moveTo(0, halfH); this.ctx.lineTo(w, halfH);
    this.ctx.stroke();
  }

  drawQuadLabel(x, y, text) {
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    this.ctx.fillRect(x - 4, y - 14, 140, 20);
    this.ctx.fillStyle = '#38bdf8';
    this.ctx.font = 'bold 11px JetBrains Mono, monospace';
    this.ctx.fillText(text, x, y);
    this.ctx.restore();
  }
}
