// oracle_ppl.cpp - exact teacher-forced perplexity for llama.cpp over a whole
// token corpus, in ONE decode pass.
//
// The previous C2 harness drove llama-server over HTTP one position at a time,
// which forced three compromises that all weaken the measurement:
//   * it read top-100 logprobs, so any position whose target fell outside the
//     oracle's top-100 was silently dropped ("unscored");
//   * n_ctx and per-request prompt reprocessing made it far too slow to score
//     the whole corpus, so it scored every 4th position up to index 300 -- 74
//     positions out of ~557;
//   * a subsample of 74 makes the reported delta a point estimate with no
//     usable confidence interval.
// Decoding the corpus directly gives every position's full 128k logit vector,
// so nothing is dropped and the delta gets a paired confidence interval.
//
//   oracle_ppl MODEL.gguf TOKENS.bin OUT.txt
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL.gguf TOKENS.bin OUT.txt\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const char * tok_path   = argv[2];
    const char * out_path   = argv[3];

    FILE * tf = fopen(tok_path, "rb");
    if (!tf) { perror(tok_path); return 1; }
    fseek(tf, 0, SEEK_END); const long nb = ftell(tf); fseek(tf, 0, SEEK_SET);
    std::vector<int32_t> ids(nb / 4);
    if (fread(ids.data(), 4, ids.size(), tf) != ids.size()) { fprintf(stderr, "short read\n"); return 1; }
    fclose(tf);
    const int n = (int) ids.size();
    if (n < 2) { fprintf(stderr, "need >= 2 tokens\n"); return 1; }

    llama_backend_init();
    if (const char * bd = getenv("LK_BACKEND_DIR")) ggml_backend_load_all_from_path(bd);
    else ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = (uint32_t) (n + 8);
    // n_ubatch sizes the compute buffer that llama.cpp RESERVES UP FRONT, and
    // reserving it for the whole corpus asked for 75504 MiB on this project's
    // 149837-token file -- more RAM than the machine has -- so the oracle died
    // in sched_reserve without scoring a single position.  Chunking is the fix
    // rather than a shorter corpus, because the whole purpose of this tool is
    // the HIGH positions and truncating would delete exactly the evidence.
    const int CHUNK = 2048;
    cp.n_batch  = (uint32_t) (n < CHUNK ? n : CHUNK);
    cp.n_ubatch = (uint32_t) (n < CHUNK ? n : CHUNK);
    cp.n_threads = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // logits for EVERY position, not just the last, produced CHUNK positions at
    // a time.  Position p predicts token p+1, so every chunk scores all of its
    // own positions except the very last position of the very last chunk; the
    // concatenation of the chunks is byte-for-byte the same scoring as the
    // single-batch version, and there is no dependence between chunks because
    // the recurrent state is carried by the context, not recomputed here.
    FILE * of = fopen(out_path, "w");
    if (!of) { perror(out_path); return 1; }
    fprintf(of, "# i nll argmax target top1ok\n");

    double nll_sum = 0.0; int cnt = 0, top1ok = 0;
    const int chunk = (int) cp.n_batch;
    llama_batch batch = llama_batch_init(chunk, 0, 1);
    for (int base = 0; base < n; base += chunk) {
        const int m = (n - base < chunk) ? (n - base) : chunk;
        for (int i = 0; i < m; ++i) {
            batch.token[i]     = ids[base + i];
            batch.pos[i]       = base + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        batch.n_tokens = m;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at %d\n", base); return 1;
        }
        for (int i = 0; i < m; ++i) {
            const int g = base + i;
            if (g + 1 >= n) break;          // no target past the end
            const float * lg = llama_get_logits_ith(ctx, i);
            if (!lg) { fprintf(stderr, "no logits at %d\n", g); return 1; }
            double m2 = -1e30; int am = 0;
            for (int v = 0; v < n_vocab; ++v) if (lg[v] > m2) { m2 = lg[v]; am = v; }
            double z = 0.0;
            for (int v = 0; v < n_vocab; ++v) z += exp((double) lg[v] - m2);
            const double lse  = m2 + log(z);
            const double nll  = -((double) lg[ids[g + 1]] - lse);
            const int    ok   = (am == ids[g + 1]);
            fprintf(of, "%d %.8f %d %d %d\n", g, nll, am, ids[g + 1], ok);
            nll_sum += nll; ++cnt; top1ok += ok;
        }
    }
    fclose(of);

    printf("ORACLE_PPL n=%d scored=%d ppl=%.6f top1_acc=%.4f n_vocab=%d\n",
           n, cnt, exp(nll_sum / cnt), (double) top1ok / cnt, n_vocab);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
