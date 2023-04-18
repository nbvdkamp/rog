# Custom glTF properties

- Lights:
    - All
        - spectrum: relative path to a csv with two columns; wavelength and relative intensity. The spectrum is normalized to unit luminance for the CIE 1931 standard observer.
    - Point
        - radius: meters as float - default: 0.0
    - Directional
        - angular_diameter: degrees as float - default: 0.0
- Materials:
    - Cauchy's law coefficients for the material, if not provided an approximation based on IOR is used.
        - cauchy_a: A coefficient as float
        - cauchy_b: B coefficient as float
- Node named 'background':
    - intensity: background light's intensity as float - default: 1.0
    - spectrum: relative path to a csv with two columns; wavelength and relative intensity. The spectrum is normalized to unit luminance for the CIE 1931 standard observer.
    - rgb: hexadecimal rgb color
