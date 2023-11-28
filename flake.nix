{
  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default =
          pkgs.mkShell {
            venvDir = ".venv";
            buildInputs = with pkgs; [
              python3
              python3Packages.venvShellHook
              python3Packages.python-lsp-server
              python3Packages.numpy
              python3Packages.pygame
              python3Packages.pytest
              python3Packages.pillow
            ];
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              pip install -r requirements.txt
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
            '';
          };
      });
}
