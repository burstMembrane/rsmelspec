use colorgrad::{self, Gradient};

/// Available colormaps.
pub enum Colormap {
    Inferno,
    Viridis,
    Plasma,
}

impl Colormap {
    pub fn to_gradient(&self) -> impl Gradient {
        match self {
            Colormap::Inferno => colorgrad::preset::inferno(),
            Colormap::Viridis => colorgrad::preset::viridis(),
            Colormap::Plasma => colorgrad::preset::plasma(),
        }
    }
}

/// Converts a normalized value (0.0 to 1.0) into an RGB color using the selected colormap.
pub fn grayscale_to_rgb(norm: f32, cmap: &Colormap) -> [u8; 3] {
    let grad = cmap.to_gradient();
    let rgba = grad.at(norm).to_rgba8();
    [rgba[0], rgba[1], rgba[2]]
}
