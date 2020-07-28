# ZLK
Quickly compute properties of von Zeipel-Lidov-Kozai oscillations in triple systems at the quadrupole expansion order and in the non-test-particle limit.

Usage: 

`python(3) ZLK.py --mode X`

where mode is 0 to use the (semi)analytic methods to compute the minimum and maximum eccentricities, and the eccentricity oscillation timescale. Initial parameters can be specified with the additional arguments

` --e0`: initial inner eccentricity
` --g0`: initial inner argument of periapsis
` --theta0`: cosine of the initial relative inclination
` --gamma`: ratio of angular momenta

For information on more options, please refer to the help: type 

`python(3) ZLK.py --help`

Note: in order to carry out numerical integrations of the equations of motion, `SecularMultiple` is required: https://github.com/hamers/secularmultiple
