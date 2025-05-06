use colorgrad::{self, Gradient};

/// Available colormaps.
pub enum Colormap {
    Inferno,
    Viridis,
    Plasma,
    Magma,
    Greys,
    Blues,
    Greens,
    Reds,
    Purples,
    Oranges,
}

impl Colormap {
    pub fn to_gradient(&self) -> impl Gradient {
        match self {
            Colormap::Inferno => colorgrad::preset::inferno(),
            Colormap::Viridis => colorgrad::preset::viridis(),
            Colormap::Plasma => colorgrad::preset::plasma(),
            Colormap::Magma => colorgrad::preset::magma(),
            Colormap::Greys => colorgrad::preset::greys(),
            Colormap::Blues => colorgrad::preset::blues(),
            Colormap::Greens => colorgrad::preset::greens(),
            Colormap::Reds => colorgrad::preset::reds(),
            Colormap::Purples => colorgrad::preset::purples(),
            Colormap::Oranges => colorgrad::preset::oranges(),
        }
    }
    pub fn from_name(name: &str) -> Option<Colormap> {
        match name {
            "inferno" => Some(Colormap::Inferno),
            "viridis" => Some(Colormap::Viridis),
            "plasma" => Some(Colormap::Plasma),
            "magma" => Some(Colormap::Magma),
            "greys" => Some(Colormap::Greys),
            "blues" => Some(Colormap::Blues),
            "greens" => Some(Colormap::Greens),
            "reds" => Some(Colormap::Reds),
            "purples" => Some(Colormap::Purples),
            "oranges" => Some(Colormap::Oranges),
            _ => None,
        }
    }
}

/// Converts a normalized value (0.0 to 1.0) into an RGB color using the selected colormap.
pub fn grayscale_to_rgb(norm: f32, cmap: &Colormap) -> [u8; 3] {
    let grad = cmap.to_gradient();
    let rgba = grad.at(norm).to_rgba8();
    [rgba[0], rgba[1], rgba[2]]
}

/// Precomputes 256 RGB colours for the given colormap.
pub fn precompute_colormap(cmap: &Colormap) -> Vec<[u8; 3]> {
    let grad = cmap.to_gradient();
    (0..256)
        .map(|i| {
            let rgba = grad.at(i as f32 / 255.0).to_rgba8();
            [rgba[0], rgba[1], rgba[2]]
        })
        .collect()
}
