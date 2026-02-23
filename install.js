module.exports = {
  run: [
    // Install requirements first
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt",
        ],
      }
    },
    // Then install torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          // xformers: true   // uncomment this line if your project requires xformers
        }
      }
    }
  ]
}
