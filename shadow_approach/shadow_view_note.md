# A Measured Formulation of the Shadow View of Collatz

## A more careful description

A useful way to think about the Collatz map is to separate the orbit into two layers:

1. the parity pattern of the terms;
2. the arithmetic values that carry that pattern.

If one records the orbit as a sequence of even/odd decisions, the resulting binary word can be viewed as a symbolic encoding of the dynamics. In the accelerated version of the map, where one removes all factors of 2 immediately after each odd step, the evolution of the values becomes closely tied to this parity word. At that symbolic level, the dynamics look simple: the orbit is encoded by a binary sequence, and the next state is determined by the next bit of that sequence.

This is a genuine simplification of the picture, but it is a simplification of the representation, not a solution of the problem.

## The Accelerated Collatz Formulation

This perspective uses the *accelerated* (or shortcut) Collatz map rather than the classical 3n+1 rule. In the accelerated version:

- If **n is even**: next = n / 2
- If **n is odd**: next = (3n + 1) / 2

By folding the immediate division by 2 after each 3n+1 step into a single operation, the accelerated map produces a shorter sequence in which every step corresponds directly to one bit in the parity word (0 = divide by 2, 1 = apply (3n+1)/2). For example, starting from 27 the classical orbit requires 111 steps, while the accelerated orbit reaches 1 in 71 steps. This formulation makes the parity shadow cleaner and more tightly coupled to the symbolic dynamics, while preserving the essential behavior of the original conjecture.

## What this perspective is

This viewpoint is best understood as a reformulation.
It highlights that:

- the Collatz map can be studied through its parity pattern;
- the parity pattern gives a symbolic description of the orbit;
- the arithmetic part of the map is pushed into the encoding of the symbolic sequence.

In this sense, the problem can be seen as a question about which binary sequences arise from ordinary integers under this encoding, and how those sequences are constrained by arithmetic.

## What this perspective is not

It is important not to overstate what this does.
It does not:

- solve the Collatz conjecture;
- prove convergence for all starting values;
- replace the classical numerical and probabilistic analysis;
- introduce a fundamentally new dynamical system.

The main point is not that the problem becomes easy in this language. The point is that the structure becomes more legible: the symbolic part is simple, while the arithmetic constraints are the part that remains difficult.

## Why it can still be useful

Even if it does not solve the conjecture, this perspective can be valuable for at least three reasons:

- it gives a compact symbolic representation of the orbit;
- it separates the simple combinatorial aspect from the arithmetic constraint;
- it connects the problem with broader ideas from symbolic dynamics and $2$-adic analysis.

So the idea is not that one has found a new mechanism for proving convergence. It is rather that one has found a different coordinate system in which the problem looks more structured and more interpretable.

## How to read the figures

The accompanying figures are intended to make the idea immediate:

- `orbit.png` shows the accelerated Collatz orbit, the numeric values visited by the sequence.
- `parity_shadow.png` shows the parity word for the same orbit, with each step recorded as `even` or `odd`.
- `combined.png` overlays the numeric orbit and the parity shadow, so the reader can see both representations on the same step axis.
- `orbit_plus_parity.png` is an illustrative example for the starting value 27, showing how the orbit and its parity word are linked.

The key visual point is:

- the numeric orbit is the arithmetic layer;
- the parity word is the symbolic shadow;
- the two describe the same underlying Collatz evolution from different perspectives.

Note: these plots show the accelerated Collatz formulation, not the classical 3n+1 sequence. For `n = 27`, the accelerated orbit reaches 1 in 71 accelerated steps, whereas the standard Collatz orbit has 111 steps and reaches a maximum of 9232.

## A short version

A serious way to state the idea is:

> The Collatz orbit can be encoded by its parity pattern. In the accelerated formulation, this encoding makes the symbolic dynamics very simple. The remaining difficulty lies in the arithmetic condition that selects which symbolic sequences actually come from ordinary integers. This is a useful reformulation and visualization of the problem, not a resolution of it.
> In the figures, the orbit values are the arithmetic layer and the parity shadow is the symbolic layer; the remaining challenge is understanding which parity words are compatible with integer orbits.

# A Measured Formulation of the Shadow View of Collatz

## A more careful description

A useful way to think about the Collatz map is to separate the orbit into two layers:

1. the parity pattern of the terms;
2. the arithmetic values that carry that pattern.

If one records the orbit as a sequence of even/odd decisions, the resulting binary word can be viewed as a symbolic encoding of the dynamics. In the accelerated version of the map, where one removes all factors of 2 immediately after each odd step, the evolution of the values becomes closely tied to this parity word. At that symbolic level, the dynamics look simple: the orbit is encoded by a binary sequence, and the next state is determined by the next bit of that sequence.

This is a genuine simplification of the picture, but it is a simplification of the representation, not a solution of the problem.

## The Accelerated Collatz Formulation

This perspective uses the *accelerated* (or shortcut) Collatz map rather than the classical $3n+1$ rule. In the accelerated version:

- If **n is even**: $\text{next} = n / 2$
- If **n is odd**: $\text{next} = (3n + 1) / 2$

By folding the immediate division by 2 after each $3n+1$ step into a single operation, the accelerated map produces a shorter sequence in which every step corresponds directly to one bit in the parity word (0 = divide by 2, 1 = apply $(3n+1)/2$). For example, starting from 27 the classical orbit requires 111 steps, while the accelerated orbit reaches 1 in 71 steps. This formulation makes the parity shadow cleaner and more tightly coupled to the symbolic dynamics, while preserving the essential behavior of the original conjecture.

## The Collatz Shadow Explorer

To actively investigate these dynamics, this repository includes a Python-based interactive desktop application (`explorer.py`). Rather than viewing static plots, the application allows you to input starting values, adjust sequence parameters, and dynamically generate up to four concurrent analytical views of the parity shadow.

### Analytical Views Included

The application provides several lenses through which to study the sequence:

- **Orbit + Shadow Track:** The fundamental view. It overlays the numeric orbit (the arithmetic layer) with the binary parity sequence (the symbolic layer) on the same step axis.
- **Rolling Density Analysis:** Displays the ratio of odd to even steps over a sliding window, highlighting localized phases of growth or decay.
- **Block Recurrence Metric:** A histogram showing the frequency of $k$-length binary blocks (e.g., how often `011` or `101` occurs), revealing combinatorial constraints.
- **2D Turtle Walk (Fractal):** Treats the parity word as directional instructions (0 = Left, 1 = Right) on a 2D plane. This maps the sequence to spatial geometry, often revealing structural symmetries hidden in the linear data.
- **Shannon Block Entropy:** Measures the informational unpredictability of the sequence over time. It highlights chaotic phases versus highly ordered phases as the orbit collapses toward 1.
- **Fast Fourier Transform (FFT):** Calculates the power spectrum of the $0-1$ signal, allowing you to check for dominant frequencies or periodic harmonics in the shadow word.
- **De Bruijn Transition Network:** A directed network graph of parity block transitions. It visualizes the chronological "flow" of the sequence, essentially mapping the pathways that drain into the standard 4-2-1 attractor.

## What this perspective is

This viewpoint is best understood as a reformulation. It highlights that:

- the Collatz map can be studied through its parity pattern;
- the parity pattern gives a symbolic description of the orbit;
- the arithmetic part of the map is pushed into the encoding of the symbolic sequence.

In this sense, the problem can be seen as a question about which binary sequences arise from ordinary integers under this encoding, and how those sequences are constrained by arithmetic.

## What this perspective is not

It is important not to overstate what this does. It does not:

- solve the Collatz conjecture;
- prove convergence for all starting values;
- replace the classical numerical and probabilistic analysis;
- introduce a fundamentally new dynamical system.

The main point is not that the problem becomes easy in this language. The point is that the structure becomes more legible: the symbolic part is simple, while the arithmetic constraints are the part that remains difficult.

## Why it can still be useful

Even if it does not solve the conjecture, this perspective can be valuable for at least three reasons:

- it gives a compact symbolic representation of the orbit;
- it separates the simple combinatorial aspect from the arithmetic constraint;
- it connects the problem with broader ideas from symbolic dynamics, signal processing, and $2$-adic analysis.

So the idea is not that one has found a new mechanism for proving convergence. It is rather that one has found a different coordinate system—aided by computational visualization—in which the problem looks more structured and more interpretable.

## Installation and Usage

To run the interactive explorer, you will need Python 3 installed along with a few standard scientific and GUI libraries.

```bash
# Clone the repository
git clone [https://github.com/yourusername/collatz-shadow.git](https://github.com/yourusername/collatz-shadow.git)
cd collatz-shadow

# Install required dependencies
pip install PyQt6 matplotlib numpy networkx

# Launch the explorer
python explorer.py
