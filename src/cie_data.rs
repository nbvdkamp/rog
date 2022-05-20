use crate::color::XYZf32;

pub const LAMBDA_MIN: f32 = 360.0;
pub const LAMBDA_MAX: f32 = 830.0;
pub const LAMBDA_RANGE: f32 = LAMBDA_MAX - LAMBDA_MIN;
pub const SAMPLES: usize = 95;

const fn xyz(x: f32, y: f32, z: f32) -> XYZf32 {
    XYZf32 { x, y, z }
}

pub static OBSERVER_1931: [XYZf32; SAMPLES] = [
    xyz(0.000129900000, 0.000003917000, 0.000606100000),
    xyz(0.000232100000, 0.000006965000, 0.001086000000),
    xyz(0.000414900000, 0.000012390000, 0.001946000000),
    xyz(0.000741600000, 0.000022020000, 0.003486000000),
    xyz(0.001368000000, 0.000039000000, 0.006450001000),
    xyz(0.002236000000, 0.000064000000, 0.010549990000),
    xyz(0.004243000000, 0.000120000000, 0.020050010000),
    xyz(0.007650000000, 0.000217000000, 0.036210000000),
    xyz(0.014310000000, 0.000396000000, 0.067850010000),
    xyz(0.023190000000, 0.000640000000, 0.110200000000),
    xyz(0.043510000000, 0.001210000000, 0.207400000000),
    xyz(0.077630000000, 0.002180000000, 0.371300000000),
    xyz(0.134380000000, 0.004000000000, 0.645600000000),
    xyz(0.214770000000, 0.007300000000, 1.039050100000),
    xyz(0.283900000000, 0.011600000000, 1.385600000000),
    xyz(0.328500000000, 0.016840000000, 1.622960000000),
    xyz(0.348280000000, 0.023000000000, 1.747060000000),
    xyz(0.348060000000, 0.029800000000, 1.782600000000),
    xyz(0.336200000000, 0.038000000000, 1.772110000000),
    xyz(0.318700000000, 0.048000000000, 1.744100000000),
    xyz(0.290800000000, 0.060000000000, 1.669200000000),
    xyz(0.251100000000, 0.073900000000, 1.528100000000),
    xyz(0.195360000000, 0.090980000000, 1.287640000000),
    xyz(0.142100000000, 0.112600000000, 1.041900000000),
    xyz(0.095640000000, 0.139020000000, 0.812950100000),
    xyz(0.057950010000, 0.169300000000, 0.616200000000),
    xyz(0.032010000000, 0.208020000000, 0.465180000000),
    xyz(0.014700000000, 0.258600000000, 0.353300000000),
    xyz(0.004900000000, 0.323000000000, 0.272000000000),
    xyz(0.002400000000, 0.407300000000, 0.212300000000),
    xyz(0.009300000000, 0.503000000000, 0.158200000000),
    xyz(0.029100000000, 0.608200000000, 0.111700000000),
    xyz(0.063270000000, 0.710000000000, 0.078249990000),
    xyz(0.109600000000, 0.793200000000, 0.057250010000),
    xyz(0.165500000000, 0.862000000000, 0.042160000000),
    xyz(0.225749900000, 0.914850100000, 0.029840000000),
    xyz(0.290400000000, 0.954000000000, 0.020300000000),
    xyz(0.359700000000, 0.980300000000, 0.013400000000),
    xyz(0.433449900000, 0.994950100000, 0.008749999000),
    xyz(0.512050100000, 1.000000000000, 0.005749999000),
    xyz(0.594500000000, 0.995000000000, 0.003900000000),
    xyz(0.678400000000, 0.978600000000, 0.002749999000),
    xyz(0.762100000000, 0.952000000000, 0.002100000000),
    xyz(0.842500000000, 0.915400000000, 0.001800000000),
    xyz(0.916300000000, 0.870000000000, 0.001650001000),
    xyz(0.978600000000, 0.816300000000, 0.001400000000),
    xyz(1.026300000000, 0.757000000000, 0.001100000000),
    xyz(1.056700000000, 0.694900000000, 0.001000000000),
    xyz(1.062200000000, 0.631000000000, 0.000800000000),
    xyz(1.045600000000, 0.566800000000, 0.000600000000),
    xyz(1.002600000000, 0.503000000000, 0.000340000000),
    xyz(0.938400000000, 0.441200000000, 0.000240000000),
    xyz(0.854449900000, 0.381000000000, 0.000190000000),
    xyz(0.751400000000, 0.321000000000, 0.000100000000),
    xyz(0.642400000000, 0.265000000000, 0.000049999990),
    xyz(0.541900000000, 0.217000000000, 0.000030000000),
    xyz(0.447900000000, 0.175000000000, 0.000020000000),
    xyz(0.360800000000, 0.138200000000, 0.000010000000),
    xyz(0.283500000000, 0.107000000000, 0.000000000000),
    xyz(0.218700000000, 0.081600000000, 0.000000000000),
    xyz(0.164900000000, 0.061000000000, 0.000000000000),
    xyz(0.121200000000, 0.044580000000, 0.000000000000),
    xyz(0.087400000000, 0.032000000000, 0.000000000000),
    xyz(0.063600000000, 0.023200000000, 0.000000000000),
    xyz(0.046770000000, 0.017000000000, 0.000000000000),
    xyz(0.032900000000, 0.011920000000, 0.000000000000),
    xyz(0.022700000000, 0.008210000000, 0.000000000000),
    xyz(0.015840000000, 0.005723000000, 0.000000000000),
    xyz(0.011359160000, 0.004102000000, 0.000000000000),
    xyz(0.008110916000, 0.002929000000, 0.000000000000),
    xyz(0.005790346000, 0.002091000000, 0.000000000000),
    xyz(0.004109457000, 0.001484000000, 0.000000000000),
    xyz(0.002899327000, 0.001047000000, 0.000000000000),
    xyz(0.002049190000, 0.000740000000, 0.000000000000),
    xyz(0.001439971000, 0.000520000000, 0.000000000000),
    xyz(0.000999949300, 0.000361100000, 0.000000000000),
    xyz(0.000690078600, 0.000249200000, 0.000000000000),
    xyz(0.000476021300, 0.000171900000, 0.000000000000),
    xyz(0.000332301100, 0.000120000000, 0.000000000000),
    xyz(0.000234826100, 0.000084800000, 0.000000000000),
    xyz(0.000166150500, 0.000060000000, 0.000000000000),
    xyz(0.000117413000, 0.000042400000, 0.000000000000),
    xyz(0.000083075270, 0.000030000000, 0.000000000000),
    xyz(0.000058706520, 0.000021200000, 0.000000000000),
    xyz(0.000041509940, 0.000014990000, 0.000000000000),
    xyz(0.000029353260, 0.000010600000, 0.000000000000),
    xyz(0.000020673830, 0.000007465700, 0.000000000000),
    xyz(0.000014559770, 0.000005257800, 0.000000000000),
    xyz(0.000010253980, 0.000003702900, 0.000000000000),
    xyz(0.000007221456, 0.000002607800, 0.000000000000),
    xyz(0.000005085868, 0.000001836600, 0.000000000000),
    xyz(0.000003581652, 0.000001293400, 0.000000000000),
    xyz(0.000002522525, 0.000000910930, 0.000000000000),
    xyz(0.000001776509, 0.000000641530, 0.000000000000),
    xyz(0.000001251141, 0.000000451810, 0.000000000000),
];

pub fn observer_1931_interp(wavelength: f32) -> XYZf32 {
    let wavelength = (wavelength - LAMBDA_MIN) * (SAMPLES - 1) as f32 / LAMBDA_RANGE;
    let offset = (wavelength as usize).min(SAMPLES - 2);
    let weight = wavelength - offset as f32;

    return (1.0 - weight) * OBSERVER_1931[offset] + weight * OBSERVER_1931[offset + 1];
}

// Normalize to luminance of 1
const fn n(x: f32) -> f32 {
    x / 10567.864005283874576
}

pub static ILLUMINANT_D65: [f32; SAMPLES] = [
    n(46.6383),
    n(49.3637),
    n(52.0891),
    n(51.0323),
    n(49.9755),
    n(52.3118),
    n(54.6482),
    n(68.7015),
    n(82.7549),
    n(87.1204),
    n(91.486),
    n(92.4589),
    n(93.4318),
    n(90.057),
    n(86.6823),
    n(95.7736),
    n(104.865),
    n(110.936),
    n(117.008),
    n(117.41),
    n(117.812),
    n(116.336),
    n(114.861),
    n(115.392),
    n(115.923),
    n(112.367),
    n(108.811),
    n(109.082),
    n(109.354),
    n(108.578),
    n(107.802),
    n(106.296),
    n(104.79),
    n(106.239),
    n(107.689),
    n(106.047),
    n(104.405),
    n(104.225),
    n(104.046),
    n(102.023),
    n(100.0),
    n(98.1671),
    n(96.3342),
    n(96.0611),
    n(95.788),
    n(92.2368),
    n(88.6856),
    n(89.3459),
    n(90.0062),
    n(89.8026),
    n(89.5991),
    n(88.6489),
    n(87.6987),
    n(85.4936),
    n(83.2886),
    n(83.4939),
    n(83.6992),
    n(81.863),
    n(80.0268),
    n(80.1207),
    n(80.2146),
    n(81.2462),
    n(82.2778),
    n(80.281),
    n(78.2842),
    n(74.0027),
    n(69.7213),
    n(70.6652),
    n(71.6091),
    n(72.979),
    n(74.349),
    n(67.9765),
    n(61.604),
    n(65.7448),
    n(69.8856),
    n(72.4863),
    n(75.087),
    n(69.3398),
    n(63.5927),
    n(55.0054),
    n(46.4182),
    n(56.6118),
    n(66.8054),
    n(65.0941),
    n(63.3828),
    n(63.8434),
    n(64.304),
    n(61.8779),
    n(59.4519),
    n(55.7054),
    n(51.959),
    n(54.6998),
    n(57.4406),
    n(58.8765),
    n(60.3125)
];

pub fn illuminant_d65_interp(wavelength: f32) -> f32 {
    let wavelength = (wavelength - LAMBDA_MIN) * (SAMPLES - 1) as f32 / LAMBDA_RANGE;
    let offset = (wavelength as usize).min(SAMPLES - 2);
    let weight = wavelength - offset as f32;

    return (1.0 - weight) * ILLUMINANT_D65[offset] + weight * ILLUMINANT_D65[offset + 1];
}