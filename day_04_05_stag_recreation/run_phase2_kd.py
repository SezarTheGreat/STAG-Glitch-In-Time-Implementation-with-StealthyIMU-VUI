import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import torch.nn.functional as F
import pandas as pd
import jsonlines

# Import original train module to reuse data preparation
import train 

class SLU_KD(train.SLU):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None, teacher_hparams=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        self.teacher_hparams = teacher_hparams
        
        # Initialize Teacher Model weights
        self.teacher_model = self.teacher_hparams["model"]
        
        # Load Teacher checkpoint
        self.teacher_hparams["checkpointer"].recover_if_possible()
        
        self.teacher_model.eval()
        self.teacher_model.to(self.device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.kd_temp = hparams["temperature_kd"]
        self.kd_alpha = hparams["alpha_kd"]

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # STUDENT FORWARD
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        encoder_out = self.hparams.enc(feats.detach())
        e_in = self.hparams.output_emb(tokens_bos)
        h, _ = self.hparams.dec(e_in, encoder_out, wav_lens)
        logits_student = self.hparams.seq_lin(h)
        p_seq_student = self.hparams.log_softmax(logits_student)

        # TEACHER FORWARD (Only during training)
        logits_teacher = None
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                feats_t = self.teacher_hparams["compute_features"](wavs)
                feats_t = self.teacher_hparams["normalize"](feats_t, wav_lens)
                encoder_out_t = self.teacher_hparams["enc"](feats_t.detach())
                e_in_t = self.teacher_hparams["output_emb"](tokens_bos)
                h_t, _ = self.teacher_hparams["dec"](e_in_t, encoder_out_t, wav_lens)
                logits_teacher = self.teacher_hparams["seq_lin"](h_t)

        # Compute outputs
        if stage == sb.Stage.TRAIN and self.batch_count % train.show_results_every != 0:
            return p_seq_student, wav_lens, logits_student, logits_teacher
        else:
            search_results = self.hparams.beam_searcher(encoder_out, wav_lens)
            if len(search_results) == 4:
                p_tokens, _, scores, _ = search_results
            else:
                p_tokens, scores = search_results
            return p_seq_student, wav_lens, p_tokens, logits_student, logits_teacher

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL + KD) given predictions and targets."""
        if stage == sb.Stage.TRAIN and self.batch_count % train.show_results_every != 0:
            p_seq_student, wav_lens, logits_student, logits_teacher = predictions
        else:
            p_seq_student, wav_lens, predicted_tokens, logits_student, logits_teacher = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        
        # Standard Seq2Seq Loss (Hard Labels)
        loss_seq = self.hparams.seq_cost(p_seq_student, tokens_eos, length=tokens_eos_lens)
        
        # Knowledge Distillation Loss (Soft Labels)
        if stage == sb.Stage.TRAIN:
            # KL Divergence between Student Log-Probs and Teacher Probs (scaled by temperature)
            student_log_probs = F.log_softmax(logits_student / self.kd_temp, dim=-1)
            teacher_probs = F.softmax(logits_teacher / self.kd_temp, dim=-1)
            
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.kd_temp ** 2)
            
            # Combine losses
            loss = (1.0 - self.kd_alpha) * loss_seq + self.kd_alpha * kd_loss
        else:
            loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (self.batch_count % train.show_results_every == 0):
            # Decode token terms to words
            predicted_semantics = [self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in predicted_tokens]
            target_semantics = [wrd.split(" ") for wrd in batch.semantics]
            
            self.log_outputs(predicted_semantics, target_semantics)

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(ids, predicted_semantics, target_semantics)
                self.cer_metric.append(ids, predicted_semantics, target_semantics)

            if stage == sb.Stage.TEST:
                # write to "predictions.jsonl"
                with jsonlines.open(self.hparams.output_folder + "/predictions.jsonl", mode="a") as writer:
                    for i in range(len(predicted_semantics)):
                        # write basic dict matching original train.py formatting trick
                        writer.write({"action": " ".join(predicted_semantics[i]), "entities": []})

        return loss

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    train.show_results_every = 200

    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from prepare import prepare_StealthyIMU

    run_on_main(
        prepare_StealthyIMU,
        kwargs={
            "data_folder": hparams["data_folder"],
            "file_name": hparams["file_name"],
            "save_folder": hparams["output_folder"],
            "train_splits": hparams["train_splits"],
            "slu_type": "direct",
            "skip_prep": hparams["skip_prep"],
            "seed": hparams["seed"],
        },
    )

    (train_set, valid_set, test_set, tokenizer) = train.dataio_prepare(hparams)

    run_on_main(hparams["pretrainer"].collect_files)
    try:
        hparams["pretrainer"].load_collected(device=run_opts["device"])
    except TypeError:
        hparams["pretrainer"].load_collected()

    # Load Teacher HParams
    print("Loading Teacher Model...")
    with open("hparams/paper_exact.yaml") as f:
        teacher_hparams = load_hyperpyyaml(f, {"seed": 1235})
    
    try:
        teacher_hparams["pretrainer"].load_collected(device=run_opts["device"])
    except TypeError:
        teacher_hparams["pretrainer"].load_collected()

    # Initialize SLU_KD Brain
    slu_brain = SLU_KD(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        teacher_hparams=teacher_hparams,
    )

    slu_brain.tokenizer = tokenizer

    print("Starting Knowledge Distillation Training!")
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    print("Starting Testing...")
    df = pd.read_csv(hparams["csv_test"])
    slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
