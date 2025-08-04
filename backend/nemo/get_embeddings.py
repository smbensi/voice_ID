import json
import OmegaConf

def get_embeddings(speaker_model, manifest_file, batch_size=1, embedding_dir='./', device='cuda'):
    test_config = OmegaConf.create(
        dict(
            manifest_filepath=manifest_file,
            sample_rate=16000,
            labels=None,
            batch_size=batch_size,
            shuffle=False,
            time_length=20,
        )
    )

    speaker_model.setup_test_data(test_config)
    speaker_model = speaker_model.to(device)
    speaker_model.eval()

    all_embs=[]
    out_embeddings = {}
           
    for test_batch in tqdm(speaker_model.test_dataloader()):
        test_batch = [x.to(device) for x in test_batch]
        audio_signal, audio_signal_len, labels, slices = test_batch
        with autocast():
            _, embs = speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
            emb_shape = embs.shape[-1]
            embs = embs.view(-1, emb_shape)
            all_embs.extend(embs.cpu().detach().numpy())
        del test_batch

    all_embs = np.asarray(all_embs)
    all_embs = embedding_normalize(all_embs)
    with open(manifest_file, 'r') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_name = '@'.join(dic['audio_filepath'].split('/')[-3:])
            out_embeddings[uniq_name] = all_embs[i]

    embedding_dir = os.path.join(embedding_dir, 'embeddings')
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir, exist_ok=True)

    prefix = manifest_file.split('/')[-1].rsplit('.', 1)[-2]

    name = os.path.join(embedding_dir, prefix)
    embeddings_file = name + '_embeddings.pkl'
    pkl.dump(out_embeddings, open(embeddings_file, 'wb'))
    print("Saved embedding files to {}".format(embedding_dir))