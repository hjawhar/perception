//! Optional live preview window using OpenCV highgui.
//!
//! Feature-gated behind the `preview` Cargo feature. When enabled, opens a
//! window that displays annotated frames with bounding boxes, labels, and
//! tracking IDs overlaid.

use crate::types::{DetectionKind, Frame, TrackedDetection};

/// Live preview window for displaying annotated frames.
pub struct PreviewWindow {
    window_name: String,
    #[allow(dead_code)]
    width: u32,
}

impl PreviewWindow {
    /// Create and open a new preview window.
    pub fn new(width: u32) -> crate::error::Result<Self> {
        let window_name = "perception".to_string();

        #[cfg(feature = "preview")]
        {
            opencv::highgui::named_window(&window_name, opencv::highgui::WINDOW_AUTOSIZE)
                .map_err(|e| crate::error::PerceptionError::OpenCv(e.to_string()))?;
        }

        Ok(Self {
            window_name,
            width,
        })
    }

    /// Display a frame with detection overlays.
    ///
    /// Returns `true` if the window should continue, `false` if ESC was pressed.
    #[cfg(feature = "preview")]
    pub fn show(
        &self,
        frame: &Frame,
        detections: &[TrackedDetection],
    ) -> crate::error::Result<bool> {
        use opencv::{core, highgui, imgproc, prelude::*};

        // Build a Mat from the frame's raw bytes.
        let mat = unsafe {
            opencv::core::Mat::new_rows_cols_with_data_unsafe_def(
                frame.height as i32,
                frame.width as i32,
                opencv::core::CV_8UC3,
                frame.data.as_ptr() as *mut std::ffi::c_void,
            )
        }
        .map_err(|e| crate::error::PerceptionError::OpenCv(e.to_string()))?;

        let mut display = mat.clone();

        for td in detections {
            let det = &td.detection;
            let color = match det.kind {
                DetectionKind::Object => core::Scalar::new(0.0, 255.0, 0.0, 0.0), // green
                DetectionKind::Face => core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // blue
                DetectionKind::Text => core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // red
            };

            let rect = core::Rect::new(
                det.bbox.x1 as i32,
                det.bbox.y1 as i32,
                (det.bbox.x2 - det.bbox.x1) as i32,
                (det.bbox.y2 - det.bbox.y1) as i32,
            );

            imgproc::rectangle(&mut display, rect, color, 2, imgproc::LINE_8, 0)
                .map_err(|e| crate::error::PerceptionError::OpenCv(e.to_string()))?;

            let label = format!(
                "[{}] {} {:.0}%",
                td.track_id,
                det.label,
                det.confidence * 100.0
            );
            imgproc::put_text(
                &mut display,
                &label,
                core::Point::new(det.bbox.x1 as i32, det.bbox.y1 as i32 - 5),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                imgproc::LINE_8,
                false,
            )
            .map_err(|e| crate::error::PerceptionError::OpenCv(e.to_string()))?;
        }

        highgui::imshow(&self.window_name, &display)
            .map_err(|e| crate::error::PerceptionError::OpenCv(e.to_string()))?;

        let key = highgui::wait_key(1)
            .map_err(|e| crate::error::PerceptionError::OpenCv(e.to_string()))?;

        // ESC key = 27
        Ok(key != 27)
    }

    /// Stub when preview feature is not enabled.
    #[cfg(not(feature = "preview"))]
    pub fn show(
        &self,
        _frame: &Frame,
        _detections: &[TrackedDetection],
    ) -> crate::error::Result<bool> {
        Ok(true)
    }
}

impl Drop for PreviewWindow {
    fn drop(&mut self) {
        #[cfg(feature = "preview")]
        {
            let _ = opencv::highgui::destroy_window(&self.window_name);
        }
    }
}
