/**
 * SmartVisionOC - Inspection Report Modal & Export Controller
 */

export class ReportModal {
  constructor(modalElement, onDownloadCallback) {
    this.modal = modalElement;
    this.currentReport = null;
    this.initCloseButtons();
  }

  initCloseButtons() {
    const closeBtns = this.modal.querySelectorAll('.close-modal-btn');
    closeBtns.forEach(btn => {
      btn.addEventListener('click', () => this.hide());
    });
    
    // Close on backdrop click
    this.modal.addEventListener('click', (e) => {
      if (e.target === this.modal) this.hide();
    });
  }

  show(reportData, fullInspectionResponse) {
    this.currentReport = {
      report: reportData,
      fullResponse: fullInspectionResponse
    };
    
    this.renderContent(reportData, fullInspectionResponse);
    this.modal.classList.add('active');
  }

  hide() {
    this.modal.classList.remove('active');
  }

  renderContent(report, full) {
    const dispositionClass = report.decision.toLowerCase();
    
    const html = `
      <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid var(--border-medium); padding-bottom: 16px; margin-bottom: 20px;">
        <div>
          <div style="display: flex; align-items: center; gap: 10px;">
            <h2 style="font-size: 18px; font-weight: 700; color: #ffffff;">CERTIFICATE OF QUALITY INSPECTION</h2>
            <span class="stage-badge">ISO 9001 / IATF 16949 COMPLIANT</span>
          </div>
          <p style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">SmartVisionOC Zero-Shot Optical Metrology Audit Trail</p>
        </div>
        <div style="text-align: right;">
          <button class="btn btn-sm close-modal-btn" style="padding: 6px 12px;">Close (Esc)</button>
        </div>
      </div>

      <!-- Disposition Header Banner -->
      <div class="decision-banner ${dispositionClass}" style="margin-bottom: 20px; padding: 16px 20px;">
        <div>
          <div style="font-size: 11px; text-transform: uppercase; font-family: var(--font-mono); letter-spacing: 1px;">FINAL DISPOSITION</div>
          <div class="decision-state-title" style="font-size: 24px;">${report.decision} - ${full.section_9_decision_engine.status_label}</div>
          <div style="font-size: 12px; opacity: 0.9; margin-top: 4px;">${full.section_9_decision_engine.decision_reason}</div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 10px; font-family: var(--font-mono); color: var(--text-muted);">CALIBRATED RISK</div>
          <div style="font-size: 26px; font-weight: 800; font-family: var(--font-mono);">${report.risk_score}</div>
        </div>
      </div>

      <!-- Meta Grid -->
      <div class="report-grid">
        <div class="report-cell">
          <div class="report-cell-label">INSPECTION ID</div>
          <div class="report-cell-value font-mono" style="font-size: 12px;">${report.inspection_id}</div>
        </div>
        <div class="report-cell">
          <div class="report-cell-label">TIMESTAMP (UTC)</div>
          <div class="report-cell-value font-mono" style="font-size: 12px;">${report.date}</div>
        </div>
        <div class="report-cell">
          <div class="report-cell-label">IDENTIFIED DEFECT</div>
          <div class="report-cell-value" style="color: #f97316;">${report.defect_type}</div>
        </div>
        <div class="report-cell">
          <div class="report-cell-label">SEVERITY / FOOTPRINT</div>
          <div class="report-cell-value font-mono">${report.defect_area}</div>
        </div>
      </div>

      <!-- Natural Language Causal Narrative -->
      <div style="margin-bottom: 20px;">
        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 8px;">
          Multi-Modal Causal Explanation
        </div>
        <div class="narrative-box">
          ${report.explanation}
        </div>
      </div>

      <!-- Actionable Recommendation -->
      <div style="margin-bottom: 24px;">
        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 8px;">
          Shop-Floor Action Directive
        </div>
        <div style="background-color: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 6px; padding: 12px; font-size: 13px; color: #38bdf8;">
          <strong>Recommendation:</strong> ${report.recommendation}
        </div>
      </div>

      <!-- Multi-Modal Visual Evidence Gallery -->
      <div style="margin-bottom: 24px;">
        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 10px;">
          Visual Telemetry Evidence Package
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
          <div class="preview-box">
            <div class="preview-box-label">Raw Component</div>
            <img src="${report.visual_evidence.raw || ''}" class="preview-img" alt="Raw" />
          </div>
          <div class="preview-box">
            <div class="preview-box-label">CLAHE Enhanced</div>
            <img src="${report.visual_evidence.clahe || ''}" class="preview-img" alt="CLAHE" />
          </div>
          <div class="preview-box">
            <div class="preview-box-label">CLIP Localization</div>
            <img src="${report.visual_evidence.heatmap || ''}" class="preview-img" alt="Heatmap" />
          </div>
          <div class="preview-box">
            <div class="preview-box-label">Depth Topography</div>
            <img src="${report.visual_evidence.depth || ''}" class="preview-img" alt="Depth" />
          </div>
        </div>
      </div>

      <!-- Audit Provenance & Actions -->
      <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border-subdued); padding-top: 16px;">
        <div style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">
          SHA-256: ${full.section_1_acquisition.sha256_hash.substring(0, 32)}...
          <br>
          <span style="color: #fbbf24;">[${full.provenance?.inspection_type === 'USER_UPLOADED' ? 'USER UPLOAD: AI INFERENCE REQUIRED' : 'PROTOTYPE SIMULATION MODE: READY FOR PYTORCH MODEL HOOK'}]</span>
        </div>
        <div style="display: flex; gap: 10px;">
          <button id="downloadJsonBtn" class="btn">Download Audit JSON</button>
          <button id="printCertBtn" class="btn btn-primary">Print Certificate</button>
        </div>
      </div>
    `;

    const container = this.modal.querySelector('.modal-body-container');
    if (container) {
      container.innerHTML = html;

      // Attach event listeners
      container.querySelector('.close-modal-btn').addEventListener('click', () => this.hide());
      
      const printBtn = container.querySelector('#printCertBtn');
      if (printBtn) {
        printBtn.addEventListener('click', () => window.print());
      }

      const downloadBtn = container.querySelector('#downloadJsonBtn');
      if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
          const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(full, null, 2));
          const dlAnchorElem = document.createElement('a');
          dlAnchorElem.setAttribute("href", dataStr);
          dlAnchorElem.setAttribute("download", `${report.inspection_id}_audit_package.json`);
          dlAnchorElem.click();
        });
      }
    }
  }
}
