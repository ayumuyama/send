def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32, norm=196):
    fr_tmp = max_fr*norm/np.sum(dataset[i][0])
    fr = fr_tmp*np.repeat(np.expand_dims(dataset[i][0], axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time,784) < fr*dt, 1, 0)
    input_spikes = input_spikes.astype(np.uint8)
    return input_spikes
