build: fmt
    cd web && tailwindcss --input src/main.css --output ../dist/main.css
    cd web && elm make src/Main.elm --output ../dist/main.js --optimize
    bunx --bun uglify-js dist/main.js --compress 'pure_funcs=[F2,F3,F4,F5,F6,F7,F8,F9,A2,A3,A4,A5,A6,A7,A8,A9],pure_getters,keep_fargs=false,unsafe_comps,unsafe' | bunx --bun uglify-js --mangle --output dist/main.min.js

lint: fmt
    uvx ruff check --fix .

fmt:
    uvx ruff format --preview .
    fd -e elm | xargs elm-format --yes
