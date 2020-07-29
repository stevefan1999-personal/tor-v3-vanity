# tor-v3-vanity
A TOR v3 vanity url generator designed to run on an nvidia GPU.

Disclaimer: This project is brand new and hasn't been thoroughly vetted.
Please report any bugs you find [here](https://github.com/dr-bonez/tor-v3-vanity/issues).

For now, the program is designed to use a single GPU with a predefined number of threads and blocks.
I will make this configurable soon.

## Installation

- [Install Rust](https://rustup.rs)
- [Install Cuda](https://developer.nvidia.com/cuda-downloads)
- `cargo install ptx-linker`
- `cargo install tor-v3-vanity`

## Usage

- Create output dir
  - `mkdir mykeys`
- Run `t3v`
  - `t3v --dst mykeys/ myprefix`
- Use the resulting file as your `hs_ed25519_secret_key`
  - `cat mykeys/myprefixwhatever.onion > /var/lib/tor/hidden_service/hs_ed25519_secret_key`
