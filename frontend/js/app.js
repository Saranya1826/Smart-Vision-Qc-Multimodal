/**
 * SmartVisionOC - Main Application Controller
 * Orchestrates the 10-stage zero-shot inspection dashboard.
 */

import { PRESET_SPECIMENS } from './presets.js';
import { CanvasViewer } from './canvasViewer.js';
import { ReportModal } from './reportModal.js';

class SmartVisionApp {
  constructor() {
    this.currentSpecimenId = "FLG-CRK-01";
    this.uploadedImageBase64 = null;
    this.inspectionData = null;
    this.history = [];
    
    this.config = {
      enable_clahe: true,
      clahe_clip_limit: 2.0,
      pass_threshold: 0.25,
      reject_threshold: 0.65,
      uncertainty_tolerance: 0.40
    };

    this.isRecordingVoice = false;
    this.speechRecognition = null;

    this.initElements();
    this.initCanvas();
    this.initPresets();
    this.initEvents();
    this.initVoiceInput();

    // Initial inspection run on startup
    this.runInspection();
  }

  initElements() {
    this.canvasEl = document.getElementById('inspectionCanvas');
    this.coordDisplayEl = document.getElementById('coordDisplay');
    this.reportModalEl = document.getElementById('reportModal');
    this.settingsModalEl = document.getElementById('settingsModal');
    this.cameraModalEl = document.getElementById('cameraModal');
    this.historyModalEl = document.getElementById('historyModal');
    
    this.runBtn = document.getElementById('runInspectionBtn');
    this.reportHeaderBtn = document.getElementById('viewReportHeaderBtn');
    this.fileInput = document.getElementById('imageFileInput');
    this.dropzone = document.getElementById('imageDropzone');
    this.alphaSlider = document.getElementById('alphaSlider');
    this.alphaDisplay = document.getElementById('alphaDisplay');
    
    // Operator Input Elements
    this.operatorNotesInput = document.getElementById('operatorNotesInput');
    this.voiceRecordBtn = document.getElementById('voiceRecordBtn');
    this.voiceStatus = document.getElementById('voiceStatus');
    this.flagToolWear = document.getElementById('flagToolWear');
    this.flagCriticalSeal = document.getElementById('flagCriticalSeal');
    this.flagQuenching = document.getElementById('flagQuenching');
    this.flagBatchRecheck = document.getElementById('flagBatchRecheck');
  }

  initCanvas() {
    this.canvasViewer = new CanvasViewer(this.canvasEl, this.coordDisplayEl);
    this.reportModal = new ReportModal(this.reportModalEl);

    // View Mode Tabs
    const viewTabs = document.querySelectorAll('.tab-btn');
    viewTabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        viewTabs.forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');
        const mode = e.target.getAttribute('data-mode');
        this.canvasViewer.setViewMode(mode);
      });
    });

    // Alpha Slider
    if (this.alphaSlider) {
      this.alphaSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        if (this.alphaDisplay) this.alphaDisplay.textContent = `${Math.round(val * 100)}%`;
        this.canvasViewer.setAlpha(val);
      });
    }
  }

  initPresets() {
    const container = document.getElementById('specimenPresetsContainer');
    if (!container) return;

    container.innerHTML = '';
    PRESET_SPECIMENS.forEach(spec => {
      const pill = document.createElement('div');
      pill.className = `specimen-pill ${spec.id === this.currentSpecimenId ? 'active' : ''}`;
      pill.setAttribute('data-id', spec.id);
      pill.innerHTML = `
        <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
          <strong style="color: #ffffff;">${spec.name}</strong>
          <span style="font-size: 9px; font-family: var(--font-mono); color: ${spec.badgeColor}; border: 1px solid ${spec.badgeColor}; padding: 1px 4px; border-radius: 3px;">${spec.tag}</span>
        </div>
        <span style="color: var(--text-muted); font-size: 10px; margin-top: 2px;">${spec.location}</span>
      `;
      pill.addEventListener('click', () => {
        document.querySelectorAll('.specimen-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        this.currentSpecimenId = spec.id;
        this.uploadedImageBase64 = null; // Clear custom upload
        this.resetInspectionUI();
        this.runInspection();
      });
      container.appendChild(pill);
    });
  }

  initEvents() {
    // Run inspection trigger
    this.runBtn.addEventListener('click', () => this.runInspection());

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.code === 'Space') {
        e.preventDefault();
        this.runInspection();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        this.openReport();
      }
    });

    // File Upload Handler
    this.fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) this.handleFile(file);
    });

    // Drag & Drop
    this.dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.dropzone.classList.add('drag-over');
    });
    this.dropzone.addEventListener('dragleave', () => {
      this.dropzone.classList.remove('drag-over');
    });
    this.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.dropzone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) this.handleFile(file);
    });

    // Camera Capture Modal
    const openCamBtn = document.getElementById('openCameraBtn');
    if (openCamBtn) {
      openCamBtn.addEventListener('click', () => {
        this.cameraModalEl.classList.add('active');
      });
    }

    const closeCamBtn = document.getElementById('closeCameraModalBtn');
    if (closeCamBtn) {
      closeCamBtn.addEventListener('click', () => {
        this.cameraModalEl.classList.remove('active');
      });
    }

    const captureSimCamBtn = document.getElementById('captureSimulatedCamBtn');
    if (captureSimCamBtn) {
      captureSimCamBtn.addEventListener('click', () => {
        this.cameraModalEl.classList.remove('active');
        this.currentSpecimenId = "FLG-CRK-01";
        this.uploadedImageBase64 = null;
        this.runInspection();
      });
    }

    // Reports Nav Trigger
    const navReports = document.getElementById('navReports');
    if (navReports) {
      navReports.addEventListener('click', () => this.openReport());
    }

    // New Inspection Nav Trigger
    const navNewInspection = document.getElementById('navNewInspection');
    if (navNewInspection) {
      navNewInspection.addEventListener('click', () => {
        this.fileInput.click();
      });
    }

    // History Nav Trigger
    const navHistory = document.getElementById('navHistory');
    if (navHistory) {
      navHistory.addEventListener('click', () => {
        this.openHistoryModal();
      });
    }

    // Settings Nav Trigger
    const navSettings = document.getElementById('navSettings');
    if (navSettings) {
      navSettings.addEventListener('click', () => {
        this.settingsModalEl.classList.add('active');
      });
    }

    const closeSettingsBtn = document.getElementById('closeSettingsBtn');
    if (closeSettingsBtn) {
      closeSettingsBtn.addEventListener('click', () => {
        this.settingsModalEl.classList.remove('active');
      });
    }

    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    if (saveSettingsBtn) {
      saveSettingsBtn.addEventListener('click', () => {
        this.config.pass_threshold = parseFloat(document.getElementById('settingPassThreshold').value);
        this.config.reject_threshold = parseFloat(document.getElementById('settingRejectThreshold').value);
        this.config.clahe_clip_limit = parseFloat(document.getElementById('settingClaheClip').value);
        this.settingsModalEl.classList.remove('active');
        this.runInspection();
      });
    }

    // Operator input change triggers live recalculation
    [this.operatorNotesInput, this.flagToolWear, this.flagCriticalSeal, this.flagQuenching, this.flagBatchRecheck].forEach(elem => {
      if (elem) {
        elem.addEventListener('input', () => this.debounceRecalculate());
        elem.addEventListener('change', () => this.debounceRecalculate());
      }
    });

    // Section 10 View Full Report Button
    const viewReportBtn = document.getElementById('viewReportBtn');
    if (viewReportBtn) {
      viewReportBtn.addEventListener('click', () => this.openReport());
    }
  }

  initVoiceInput() {
    if (!this.voiceRecordBtn) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      this.speechRecognition = new SpeechRecognition();
      this.speechRecognition.continuous = false;
      this.speechRecognition.interimResults = false;
      this.speechRecognition.lang = 'en-US';

      this.speechRecognition.onstart = () => {
        this.isRecordingVoice = true;
        this.voiceRecordBtn.style.backgroundColor = '#ef4444';
        this.voiceRecordBtn.style.color = '#ffffff';
        if (this.voiceStatus) this.voiceStatus.textContent = "Listening... Speak shop-floor notes.";
      };

      this.speechRecognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (this.operatorNotesInput) {
          this.operatorNotesInput.value = (this.operatorNotesInput.value + " " + transcript).trim();
          this.debounceRecalculate();
        }
      };

      this.speechRecognition.onend = () => {
        this.isRecordingVoice = false;
        this.voiceRecordBtn.style.backgroundColor = '';
        this.voiceRecordBtn.style.color = '';
        if (this.voiceStatus) this.voiceStatus.textContent = "Voice input idle.";
      };

      this.voiceRecordBtn.addEventListener('click', () => {
        if (this.isRecordingVoice) {
          this.speechRecognition.stop();
        } else {
          this.speechRecognition.start();
        }
      });
    } else {
      // Fallback: Simulated Industrial Voice Transcription
      this.voiceRecordBtn.addEventListener('click', () => {
        this.voiceRecordBtn.style.backgroundColor = '#f59e0b';
        if (this.voiceStatus) this.voiceStatus.textContent = "Simulating voice input...";
        setTimeout(() => {
          const sampleAudioPrompts = [
            "Operator noted mechanical chatter near bolt hole 3 during finish pass.",
            "Worn ceramic tool insert detected at spindle 2.",
            "Cosmetic water stain from cleaning bath, verify depth flatness.",
            "Conforming visual inspection, phonographic grooves intact."
          ];
          const chosen = sampleAudioPrompts[Math.floor(Math.random() * sampleAudioPrompts.length)];
          this.operatorNotesInput.value = chosen;
          this.voiceRecordBtn.style.backgroundColor = '';
          if (this.voiceStatus) this.voiceStatus.textContent = "Transcript captured.";
          this.debounceRecalculate();
        }, 1200);
      });
    }
  }

  handleFile(file) {
    if (!file) return;

    // 1. Max size validation: 25 MB
    const maxSizeBytes = 25 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      alert(`Image file size (${(file.size / (1024 * 1024)).toFixed(2)} MB) exceeds the 25 MB maximum limit.`);
      if (this.fileInput) this.fileInput.value = '';
      return;
    }

    // 2. Format validation: PNG, JPG, JPEG, BMP, WEBP
    const validExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp'];
    const lowerName = file.name.toLowerCase();
    const isValidFormat = validExtensions.some(ext => lowerName.endsWith(ext)) || file.type.startsWith('image/');

    if (!isValidFormat) {
      alert("Unable to read this image. Please upload a valid PNG, JPG, JPEG, BMP, or WEBP image.");
      if (this.fileInput) this.fileInput.value = '';
      return;
    }

    // 3. Clear previous preset/inspection state
    this.currentSpecimenId = null;
    this.uploadedImageBase64 = null;
    this.inspectionData = null;

    // Deselect preset pills
    document.querySelectorAll('.specimen-pill').forEach(p => p.classList.remove('active'));

    // Reset UI state to clear old predictions, heatmaps, and masks
    this.resetInspectionUI();

    // 4. Read file and display immediate preview
    const reader = new FileReader();
    reader.onerror = () => {
      alert("Failed to read image file from disk. Please try again.");
    };
    reader.onload = (e) => {
      this.uploadedImageBase64 = e.target.result;
      
      // Update Acquisition preview thumbnail immediately
      const previewRaw = document.getElementById('previewRawThumb');
      if (previewRaw) previewRaw.src = this.uploadedImageBase64;
      
      // Load into canvas viewer immediately
      const img = new Image();
      img.onload = () => {
        const s1Meta = document.getElementById('s1Meta');
        if (s1Meta) {
          const sizeKb = (file.size / 1024.0).toFixed(1);
          s1Meta.innerHTML = `${img.naturalWidth} &times; ${img.naturalHeight} px | ${sizeKb} KB | [USER UPLOAD]`;
        }
        if (this.canvasViewer) {
          this.canvasViewer.reset(this.uploadedImageBase64);
        }
      };
      img.src = this.uploadedImageBase64;

      // Run inference on the actual uploaded image
      this.runInspection();
    };
    reader.readAsDataURL(file);

    // Reset input so selecting the same file again triggers change event
    if (this.fileInput) this.fileInput.value = '';
  }

  resetInspectionUI() {
    // Reset Canvas
    if (this.canvasViewer) {
      this.canvasViewer.reset();
    }

    // Reset Header Status Badge
    const statusBadge = document.querySelector('.header-status-badge');
    if (statusBadge) {
      statusBadge.innerHTML = `
        <div class="pulse-dot" style="background-color: #38bdf8;"></div>
        <span>SYSTEM STATUS: RUNNING INSPECTION...</span>
      `;
    }

    // Reset Preprocessing thumbs and metrics
    const s2Orig = document.getElementById('s2OriginalThumb');
    if (s2Orig) s2Orig.src = '';
    const s2Enh = document.getElementById('s2EnhancedThumb');
    if (s2Enh) s2Enh.src = '';
    const s2Quality = document.getElementById('s2QualityBadge');
    if (s2Quality) s2Quality.textContent = 'Processing image...';
    ['s2FocusVal', 's2BrightnessVal', 's2ContrastVal', 's2NoiseVal'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '---';
    });

    // Reset Defect ID
    const s3Defect = document.getElementById('s3PredictedDefect');
    if (s3Defect) {
      s3Defect.textContent = 'Analyzing...';
      s3Defect.style.fontSize = '14px';
      s3Defect.style.color = '#38bdf8';
    }
    const s3Conf = document.getElementById('s3ClipConfidence');
    if (s3Conf) s3Conf.textContent = '---%';
    const s3Heatmap = document.getElementById('s3HeatmapThumb');
    if (s3Heatmap) s3Heatmap.src = '';
    const s3Bars = document.getElementById('s3ConfidenceBars');
    if (s3Bars) s3Bars.innerHTML = '<div style="color: var(--text-muted); font-size: 11px;">Extracting semantic features...</div>';

    // Reset Segmentation
    const s4Mask = document.getElementById('s4MaskThumb');
    if (s4Mask) s4Mask.src = '';
    const s4Area = document.getElementById('s4DefectArea');
    if (s4Area) s4Area.textContent = '---';
    const s4Bbox = document.getElementById('s4BoundingBox');
    if (s4Bbox) s4Bbox.textContent = '---';
    const s4Centroid = document.getElementById('s4Centroid');
    if (s4Centroid) s4Centroid.textContent = '---';
    const s4Conf = document.getElementById('s4SamConfidence');
    if (s4Conf) s4Conf.textContent = '---%';

    // Reset Spatial
    const s5Depth = document.getElementById('s5DepthThumb');
    if (s5Depth) s5Depth.src = '';
    ['s5RelativeDepth', 's5Location', 's5SurfaceProfile'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '---';
    });

    // Reset Fusion
    ['s7VisualScore', 's7OperatorScore', 's7AgreementScore', 's7CombinedConfidence'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '---';
    });

    // Reset Risk
    ['s8ClipScore', 's8SamScore', 's8AreaScore', 's8AgreeScore', 's8UncertaintyPenalty'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '---';
    });
    const s8RiskVal = document.getElementById('s8RiskValue');
    if (s8RiskVal) s8RiskVal.textContent = '0.00';
    const s8RiskLevel = document.getElementById('s8RiskLevel');
    if (s8RiskLevel) s8RiskLevel.textContent = 'PROCESSING';
    const s8RiskBar = document.getElementById('s8RiskBar');
    if (s8RiskBar) s8RiskBar.style.width = '0%';

    // Reset Decision Banner
    const banner = document.getElementById('decisionBanner');
    if (banner) banner.className = 'decision-banner review';
    const dState = document.getElementById('decisionStateText');
    if (dState) dState.textContent = 'ANALYZING';
    const dReason = document.getElementById('decisionReasonText');
    if (dReason) dReason.textContent = 'Evaluating multi-modal sensor telemetry...';

    // Reset Report Preview
    ['reportInspectionId', 'reportDate', 'reportDefectType', 'reportLocation', 'reportArea', 'reportConfidence', 'reportDecision', 'reportRisk'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '---';
    });
    const repExpl = document.getElementById('reportExplanation');
    if (repExpl) repExpl.textContent = 'Awaiting inspection execution...';
    const repRec = document.getElementById('reportRecommendation');
    if (repRec) repRec.textContent = '---';
  }

  debounceRecalculate() {
    clearTimeout(this._debounceTimer);
    this._debounceTimer = setTimeout(() => {
      this.runInspection();
    }, 400);
  }

  async runInspection() {
    this.runBtn.disabled = true;
    this.runBtn.innerHTML = `
      <div class="pulse-dot" style="background-color: #ffffff;"></div>
      <span>Running inspection...</span>
    `;

    const payload = {
      specimen_id: this.uploadedImageBase64 ? null : this.currentSpecimenId,
      image_base64: this.uploadedImageBase64,
      operator_input: {
        notes: this.operatorNotesInput ? this.operatorNotesInput.value : "",
        voice_transcript: "",
        tool_wear_alert: this.flagToolWear ? this.flagToolWear.checked : false,
        critical_sealing_surface: this.flagCriticalSeal ? this.flagCriticalSeal.checked : false,
        quenching_anomaly: this.flagQuenching ? this.flagQuenching.checked : false,
        batch_recheck: this.flagBatchRecheck ? this.flagBatchRecheck.checked : false
      },
      config: this.config
    };

    try {
      const response = await fetch('/api/v1/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        let errorMsg = response.statusText;
        try {
          const errData = await response.json();
          if (errData.detail) errorMsg = errData.detail;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      const data = await response.json();
      this.inspectionData = data;
      this.history.unshift({
        id: data.inspection_id,
        timestamp: data.timestamp,
        defect: data.section_3_defect_identification.predicted_defect,
        disposition: data.section_9_decision_engine.disposition,
        risk: data.section_8_risk_scoring.weighted_risk_score
      });
      
      this.renderAllSections(data);
      await this.canvasViewer.loadInspectionData(data);
    } catch (err) {
      console.error("Inspection Pipeline Error:", err);
      alert(`Inspection Pipeline Error: ${err.message}`);
    } finally {
      this.runBtn.disabled = false;
      this.runBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        <span>Run Inspection (Space)</span>
      `;
    }
  }

  renderAllSections(data) {
    // -------------------------------------------------------------
    // HEADER STATUS BADGE (Requirement 18: Model Status Section)
    // -------------------------------------------------------------
    const statusBadge = document.querySelector('.header-status-badge');
    if (statusBadge) {
      const modeText = data.provenance?.execution_mode || "CLIP: FALLBACK | SAM 2: FALLBACK | DEPTH: FALLBACK";
      statusBadge.innerHTML = `
        <div class="pulse-dot"></div>
        <span>SYSTEM STATUS: ${modeText}</span>
      `;
    }

    const isUploaded = data.provenance?.inspection_type === "USER_UPLOADED";

    // -------------------------------------------------------------
    // SECTION 1: IMAGE ACQUISITION
    // -------------------------------------------------------------
    const s1 = data.section_1_acquisition;
    const previewRaw = document.getElementById('previewRawThumb');
    if (previewRaw) previewRaw.src = s1.image_base64;
    const s1Meta = document.getElementById('s1Meta');
    if (s1Meta) {
      const modeLabel = isUploaded ? "USER UPLOAD" : "DEMO PRESET";
      s1Meta.innerHTML = `${s1.original_dimensions[0]} &times; ${s1.original_dimensions[1]} px | ${s1.file_size_kb} KB | [${modeLabel}]`;
    }

    // -------------------------------------------------------------
    // SECTION 2: PREPROCESSING
    // -------------------------------------------------------------
    const s2 = data.section_2_preprocessing;
    const s2Orig = document.getElementById('s2OriginalThumb');
    if (s2Orig) s2Orig.src = s2.original_image_base64;
    const s2Enh = document.getElementById('s2EnhancedThumb');
    if (s2Enh) s2Enh.src = s2.enhanced_image_base64;

    const s2Quality = document.getElementById('s2QualityBadge');
    if (s2Quality) s2Quality.textContent = s2.quality_check;
    const s2Focus = document.getElementById('s2FocusVal');
    if (s2Focus) s2Focus.textContent = s2.focus_score;
    const s2Brightness = document.getElementById('s2BrightnessVal');
    if (s2Brightness) s2Brightness.textContent = `${s2.brightness_score}%`;
    const s2Contrast = document.getElementById('s2ContrastVal');
    if (s2Contrast) s2Contrast.textContent = s2.contrast_score;
    const s2Noise = document.getElementById('s2NoiseVal');
    if (s2Noise) s2Noise.textContent = `${s2.noise_level_snr} dB`;

    // -------------------------------------------------------------
    // SECTION 3: DEFECT IDENTIFICATION (Requirement 2, 3, 4, 5)
    // -------------------------------------------------------------
    const s3 = data.section_3_defect_identification;
    const s3Defect = document.getElementById('s3PredictedDefect');
    const s3Conf = document.getElementById('s3ClipConfidence');
    const s3HeatmapThumb = document.getElementById('s3HeatmapThumb');
    const s3BarContainer = document.getElementById('s3ConfidenceBars');

    const s3Badge = document.getElementById('s3StageBadge');
    if (s3Badge) s3Badge.textContent = `CLIP: ${s3.model_status || 'FALLBACK'}`;
    if (s3Defect) {
      s3Defect.textContent = s3.predicted_defect;
      s3Defect.style.fontSize = '16px';
      s3Defect.style.color = s3.predicted_defect === 'Normal' ? '#10b981' : '#f97316';
    }
    if (s3Conf) s3Conf.textContent = `${(s3.clip_confidence * 100).toFixed(1)}%`;
    if (s3HeatmapThumb) s3HeatmapThumb.src = s3.heatmap_base64;

    // Render Zero-Shot Similarity Spectrum
    if (s3BarContainer && s3.similarity_scores) {
      s3BarContainer.innerHTML = '';
      for (const [defectClass, score] of Object.entries(s3.similarity_scores)) {
        const isTop = defectClass === s3.predicted_defect;
        const pct = Math.round(score * 100);
        const item = document.createElement('div');
        item.className = 'class-progress-item';
        item.innerHTML = `
          <div class="class-progress-header">
            <span style="color: ${isTop ? (defectClass === 'Normal' ? '#10b981' : '#f97316') : 'var(--text-secondary)'}; font-weight: ${isTop ? '700' : '500'};">${defectClass}</span>
            <span class="font-mono" style="color: ${isTop ? '#ffffff' : 'var(--text-muted)'};">${pct}%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill ${isTop ? 'top-defect' : ''}" style="width: ${pct}%; background-color: ${isTop && defectClass === 'Normal' ? '#10b981' : ''}"></div>
          </div>
        `;
        s3BarContainer.appendChild(item);
      }
    }

    // -------------------------------------------------------------
    // SECTION 4: SEGMENTATION (Requirement 6, 7, 8, 9, 10)
    // -------------------------------------------------------------
    const s4 = data.section_4_segmentation;
    const s4Badge = document.getElementById('s4StageBadge');
    if (s4Badge) s4Badge.textContent = `SAM 2: ${s4.model_status || 'FALLBACK'}`;
    const s4MaskThumb = document.getElementById('s4MaskThumb');
    if (s4MaskThumb) s4MaskThumb.src = s4.mask_base64;
    
    // Area: XXXX px (X.XX%) and Physical area note
    const s4Area = document.getElementById('s4DefectArea');
    if (s4Area) {
      if (s3.predicted_defect === 'Normal' || s4.defect_area_px === 0) {
        s4Area.innerHTML = `<span>0 px (0.00%)</span><br><span style="font-size: 9px; color: var(--text-muted);">Physical area: Conforming</span>`;
      } else {
        const pct = ((s4.defect_area_px / (s1.original_dimensions[0] * s1.original_dimensions[1])) * 100).toFixed(2);
        s4Area.innerHTML = `<span>${s4.defect_area_px.toLocaleString()} px (${pct}%)</span><br><span style="font-size: 9px; color: #f59e0b;">Physical area: Not calibrated</span>`;
      }
    }

    const s4Bbox = document.getElementById('s4BoundingBox');
    if (s4Bbox) {
      if (s3.predicted_defect === 'Normal' || !s4.bounding_box || s4.bounding_box[2] === 0) {
        s4Bbox.textContent = '[None]';
      } else {
        s4Bbox.textContent = `[${s4.bounding_box.join(', ')}]`;
      }
    }

    const s4Centroid = document.getElementById('s4Centroid');
    if (s4Centroid) {
      if (s3.predicted_defect === 'Normal' || !s4.centroid || (s4.centroid[0] === 0 && s4.centroid[1] === 0)) {
        s4Centroid.textContent = 'No defect region';
      } else {
        s4Centroid.textContent = `X: ${s4.centroid[0]} px, Y: ${s4.centroid[1]} px`;
      }
    }

    const s4Conf = document.getElementById('s4SamConfidence');
    if (s4Conf) s4Conf.textContent = `${(s4.sam_confidence * 100).toFixed(1)}%`;

    // -------------------------------------------------------------
    // SECTION 5: SPATIAL ANALYSIS (Requirement 11, 12, 13)
    // -------------------------------------------------------------
    const s5 = data.section_5_spatial_analysis;
    const s5Badge = document.getElementById('s5StageBadge');
    if (s5Badge) s5Badge.textContent = `DEPTH: ${s5.model_status || 'FALLBACK'}`;
    const s5DepthThumb = document.getElementById('s5DepthThumb');
    if (s5DepthThumb) s5DepthThumb.src = s5.depth_map_base64;
    
    const s5RelDepth = document.getElementById('s5RelativeDepth');
    if (s5RelDepth) {
      if (s3.predicted_defect === 'Normal') {
        s5RelDepth.textContent = '+0.00 (Planar Nominal)';
      } else {
        const sign = s5.relative_depth_mm >= 0 ? '+' : '';
        s5RelDepth.textContent = `${sign}${s5.relative_depth_mm.toFixed(2)} (Normalized)`;
      }
    }

    const s5Location = document.getElementById('s5Location');
    if (s5Location) s5Location.textContent = s5.location;
    const s5Profile = document.getElementById('s5SurfaceProfile');
    if (s5Profile) s5Profile.textContent = s5.surface_profile;

    // -------------------------------------------------------------
    // SECTION 7: EVIDENCE FUSION (Requirement 14)
    // -------------------------------------------------------------
    const s7 = data.section_7_evidence_fusion;
    const s7Vis = document.getElementById('s7VisualScore');
    if (s7Vis) {
      const wVis = s7.fusion_weights?.visual || 0.35;
      s7Vis.textContent = `${(s7.visual_evidence * 100).toFixed(1)}% (wt: ${wVis})`;
    }
    const s7Op = document.getElementById('s7OperatorScore');
    if (s7Op) {
      const wOp = s7.fusion_weights?.operator || 0.15;
      s7Op.textContent = `${(s7.operator_evidence * 100).toFixed(1)}% (wt: ${wOp})`;
    }
    const s7Agree = document.getElementById('s7AgreementScore');
    if (s7Agree) s7Agree.textContent = `${(s7.evidence_agreement * 100).toFixed(1)}%`;
    const s7CombConf = document.getElementById('s7CombinedConfidence');
    if (s7CombConf) s7CombConf.textContent = `${(s7.combined_confidence * 100).toFixed(1)}%`;

    // -------------------------------------------------------------
    // SECTION 8: RISK SCORING (Requirement 15)
    // -------------------------------------------------------------
    const s8 = data.section_8_risk_scoring;
    const s8Clip = document.getElementById('s8ClipScore');
    if (s8Clip) s8Clip.textContent = `${(s8.clip_confidence * 100).toFixed(1)}%`;
    const s8Sam = document.getElementById('s8SamScore');
    if (s8Sam) s8Sam.textContent = `${(s8.sam_confidence * 100).toFixed(1)}%`;
    const s8Area = document.getElementById('s8AreaScore');
    if (s8Area) s8Area.textContent = s8.defect_area_score.toFixed(2);
    const s8Agree = document.getElementById('s8AgreeScore');
    if (s8Agree) s8Agree.textContent = `${(s8.evidence_agreement * 100).toFixed(1)}%`;
    const s8Uncert = document.getElementById('s8UncertaintyPenalty');
    if (s8Uncert) s8Uncert.textContent = `+${(s8.uncertainty_penalty).toFixed(2)}`;
    
    const s8RiskVal = document.getElementById('s8RiskValue');
    if (s8RiskVal) s8RiskVal.textContent = s8.weighted_risk_score.toFixed(2);
    const s8RiskLevel = document.getElementById('s8RiskLevel');
    if (s8RiskLevel) {
      s8RiskLevel.textContent = s8.risk_level.toUpperCase();
      if (s8.risk_level.toLowerCase() === 'nominal' || s8.risk_level.toLowerCase() === 'low') {
        s8RiskLevel.style.color = '#10b981';
      } else if (s8.risk_level.toLowerCase() === 'moderate') {
        s8RiskLevel.style.color = '#f59e0b';
      } else {
        s8RiskLevel.style.color = '#ef4444';
      }
    }
    const s8RiskBar = document.getElementById('s8RiskBar');
    if (s8RiskBar) {
      s8RiskBar.style.width = `${Math.round(s8.weighted_risk_score * 100)}%`;
      if (s8.weighted_risk_score <= 0.25) {
        s8RiskBar.style.backgroundColor = '#10b981';
      } else if (s8.weighted_risk_score < 0.65) {
        s8RiskBar.style.backgroundColor = '#f59e0b';
      } else {
        s8RiskBar.style.backgroundColor = '#ef4444';
      }
    }

    // -------------------------------------------------------------
    // SECTION 9: DECISION ENGINE
    // -------------------------------------------------------------
    const s9 = data.section_9_decision_engine;
    const decisionBanner = document.getElementById('decisionBanner');
    const decisionText = document.getElementById('decisionStateText');
    const decisionReason = document.getElementById('decisionReasonText');
    
    if (decisionBanner) {
      decisionBanner.className = `decision-banner ${s9.disposition.toLowerCase()}`;
    }
    if (decisionText) {
      decisionText.textContent = s9.disposition;
    }
    if (decisionReason) {
      decisionReason.textContent = s9.decision_reason;
    }

    // -------------------------------------------------------------
    // SECTION 10: EXPLAINABLE INSPECTION REPORT PREVIEW (Requirement 16, 17, 22)
    // -------------------------------------------------------------
    const s10 = data.section_10_inspection_report;
    const repId = document.getElementById('reportInspectionId');
    if (repId) repId.textContent = s10.inspection_id;
    const repDate = document.getElementById('reportDate');
    if (repDate) repDate.textContent = s10.date;
    const repDefect = document.getElementById('reportDefectType');
    if (repDefect) {
      repDefect.textContent = s10.defect_type;
      repDefect.style.color = s10.defect_type === 'Normal' ? '#10b981' : '#f97316';
    }
    const repLoc = document.getElementById('reportLocation');
    if (repLoc) repLoc.textContent = s10.location;
    const repArea = document.getElementById('reportArea');
    if (repArea) repArea.textContent = s10.defect_area;
    const repConf = document.getElementById('reportConfidence');
    if (repConf) repConf.textContent = s10.confidence;
    const repRisk = document.getElementById('reportRisk');
    if (repRisk) repRisk.textContent = s10.risk_score;
    const repDec = document.getElementById('reportDecision');
    if (repDec) repDec.textContent = s10.decision;
    const repExpl = document.getElementById('reportExplanation');
    if (repExpl) repExpl.textContent = s10.explanation;
    const repRec = document.getElementById('reportRecommendation');
    if (repRec) repRec.textContent = s10.recommendation;
  }

  openReport() {
    if (this.inspectionData) {
      this.reportModal.show(this.inspectionData.section_10_inspection_report, this.inspectionData);
    }
  }

  openHistoryModal() {
    if (!this.historyModalEl) return;
    const listEl = document.getElementById('historyListContainer');
    if (listEl) {
      if (this.history.length === 0) {
        listEl.innerHTML = `<p style="color: var(--text-muted); padding: 16px;">No past inspection runs recorded.</p>`;
      } else {
        listEl.innerHTML = this.history.map(item => `
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--border-subdued); background-color: var(--bg-elevated); margin-bottom: 6px; border-radius: 4px;">
            <div>
              <div class="font-mono" style="color: #ffffff; font-weight: 600;">${item.id}</div>
              <div style="font-size: 11px; color: var(--text-muted);">${item.timestamp} | Defect: ${item.defect}</div>
            </div>
            <div style="text-align: right;">
              <span class="stage-badge" style="color: ${item.disposition === 'PASS' ? '#10b981' : (item.disposition === 'REJECT' ? '#ef4444' : '#f59e0b')}; border-color: currentColor;">
                ${item.disposition}
              </span>
              <div class="font-mono" style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">Risk: ${item.risk.toFixed(2)}</div>
            </div>
          </div>
        `).join('');
      }
    }
    this.historyModalEl.classList.add('active');
    const closeBtn = document.getElementById('closeHistoryBtn');
    if (closeBtn) closeBtn.onclick = () => this.historyModalEl.classList.remove('active');
  }
}

// Bootstrap application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.app = new SmartVisionApp();
});
