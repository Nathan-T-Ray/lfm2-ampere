// oracle_logits.cpp - dump llama.cpp's FULL vocabulary logit vector for the
// last position of a prompt, so C1a can compare distributions rather than
// argmaxes.  The HTTP API only exposes top-k logprobs; linking libllama
// directly gives all n_vocab floats.
//
//   oracle_logits MODEL.gguf OUT.bin TOKEN_ID...
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf OUT.bin TOK...\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];

    std::vector<llama_token> toks;
    for (int i = 3; i < argc; ++i) toks.push_back((llama_token) atoi(argv[i]));
    if (toks.empty()) { fprintf(stderr, "no tokens\n"); return 1; }

    llama_backend_init();
    // llama_backend_init() only initialises the CPU backend; the CUDA
    // backend lives in a separately-loaded shared object, and without it
    // the loader refuses to run at all.
    if (const char * bd = getenv("LK_BACKEND_DIR")) ggml_backend_load_all_from_path(bd);
    else ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;          // everything on the GPU, as the server does
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 8192;            // match scripts/oracle.sh
    cp.n_batch  = 2048;
    cp.n_ubatch = 512;
    cp.n_threads = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const float * logits = llama_get_logits_ith(ctx, -1);   // last position
    if (!logits) { fprintf(stderr, "no logits\n"); return 1; }

    FILE * f = fopen(out_path, "wb");
    if (!f) { perror(out_path); return 1; }
    if (fwrite(logits, sizeof(float), (size_t) n_vocab, f) != (size_t) n_vocab) {
        fprintf(stderr, "short write\n"); return 1;
    }
    fclose(f);

    int am = 0;
    for (int v = 1; v < n_vocab; ++v) if (logits[v] > logits[am]) am = v;
    printf("ORACLE n_ctx=%u n_vocab=%d n_tok=%zu argmax=%d logit=%.6f -> %s\n",
           llama_n_ctx(ctx), n_vocab, toks.size(), am, logits[am], out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
