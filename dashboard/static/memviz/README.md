# Vendored PyTorch MemoryViz viewer

Copied verbatim from PyTorch's `torch/utils/viz/` directory:

- `MemoryViz.js`
- `process_alloc_data.js`

Source: <https://github.com/pytorch/pytorch/tree/main/torch/utils/viz>

The dashboard serves these from `/api/runs/<id>/memory_viz` (see
`dashboard/server.py`). `MemoryViz.js` imports its d3 dependency from jsDelivr at
runtime, exactly like the hosted <https://docs.pytorch.org/memory_viz> page; the
relative `./process_alloc_data.js` import resolves to the sibling file here. To
refresh, re-download both files from the source URL above.
