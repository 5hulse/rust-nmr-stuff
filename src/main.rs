use ndarray::prelude::*;
use num::complex::Complex;
use std::f64::consts::PI;
use::std::fs::File;
use std::io::prelude::*;

fn make_fid(parameters: &Array<f64,Ix2>, n: usize, sw: f64, offset: f64) -> (Array<f64,Ix1>, Array<Complex<f64>,Ix1>) {
    // Extract amplitudes, phases, frequencies, and dampings
    let amp = parameters.slice(s![.., 0])
                        .to_owned();

    let phase = parameters.slice(s![.., 1])
                          .to_owned();

    let freq = parameters.slice(s![.., 2])
                         .to_owned()
                         .mapv(|x| x - offset);

    let damp = parameters.slice(s![.., 3])
                         .to_owned();

    // Time-points
    let tp = Array::linspace(0., n as f64/sw, n);

    // Complex amplitudes (α)
    let alpha = amp.mapv(|x| Complex::new(x, 0.)) * phase.mapv(|x| Complex::new(0., x).exp());

    // Generate Vandermonde matrix Z using an outer product
    let z = tp.mapv(|x| Complex::new(x, 0.)) // Make time-points complex
              .insert_axis(Axis(1)) // [N] -> [N, 1]
              .dot::<Array<Complex<f64>,Ix2>>(
                  &(freq.mapv(|x| Complex::new(0., 2.*PI*x)) -
                    damp.mapv(|x| Complex::new(x, 0.))) // 2πif - η
                        .insert_axis(Axis(0)) // [M] -> [1, M]
              ).mapv(|x| x.exp()) // element-wise exponential
               .to_owned();

    // fid = Zα
    let fid = z.dot::<Array<Complex<f64>,Ix1>>(&alpha);

    (tp, fid)
}

fn main() -> std::io::Result<()> {
    let params = array![[1., 0., 3., 0.1], [1., 0., -2., 0.1]];
    let n = 512;
    let sw = 20.;
    let offset = 0.;
    let (_, fid) = make_fid(&params, n, sw, offset);

    // write FID to textfile
    let mut file = File::create("fid.txt")?;
    for pt in fid.iter() {
        file.write(format!("{} + {}i\n", pt.re, pt.im).as_bytes()).expect("failed to write");
    }
    Ok(())
}
