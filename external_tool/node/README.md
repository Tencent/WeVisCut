# Node.js runtime

The HyperFrames CLI is a Node program and requires **Node >= 22**.

If your machine already has a suitable Node, skip this directory entirely and
point the config at it:

```yaml
services:
  hyperframes:
    node_bin_dir: /usr/local/bin
```

or `export WEDIRECTOR_NODE_BIN=/usr/local/bin`.

To vendor a runtime here instead (what `scripts/setup_external_tools.sh` does):

```bash
cd external_tool
curl -L -o node.tar.xz https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz
mkdir -p node && tar xf node.tar.xz -C node --strip-components=1
rm node.tar.xz
./node/bin/node --version   # expect v22.x or newer
```

The resulting layout is `external_tool/node/bin/{node,npm,npx}`, which is what
`services.hyperframes.node_bin_dir` points at by default.

Adjust the URL for your platform (`darwin-arm64`, `linux-arm64`, ...). Any Node
22 or newer distribution works; the exact patch version does not matter.
