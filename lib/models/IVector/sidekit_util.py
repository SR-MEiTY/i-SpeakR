"""
Created on 6th September 2022

@authors: 
    Dr. Joyanta Basu, Madhab Pal, Soma Khan, CDAC Kolkata
@collaborators: 
    Jagabandhu Mishra, Ph.D. Scholar, Dept. of EE, IIT Dharwad
    Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad

Ref: SIDEKIT is an open source package for Speaker and Language recognition (https://projets-lium.univ-lemans.fr/sidekit/tutorial/sre10_iv.html)
"""



import os
import numpy
import sidekit
from lib.models.IVector.config import config
from lib.models.IVector.statserver import StatServer


def id_map_list(ubm_feat_dir):
    right_list = []
    left_list = []
    print(f'ubm_feat_dir={ubm_feat_dir}')

    # for h5_file in os.listdir(ubm_feat_dir):
    #     if h5_file.find('.h5') == -1:
    #         continue
    #     show_name = h5_file.split('.h5')[0]
    #     right_list.append(show_name)
    #     left_list.append(show_name.split('.')[0])

    for npy_file in os.listdir(ubm_feat_dir):
        print(f'npy_file={npy_file}')
        if npy_file.find('.npy') == -1:
            continue
        show_name = npy_file.split('.npy')[0]
        print(f'show_name={show_name}')
        right_list.append(show_name)
        left_list.append(show_name.split('.')[0])

    return left_list, right_list


def get_id_map(leftlist, rightlist):
    idmap = sidekit.IdMap()
    idmap.leftids = numpy.array(leftlist)
    idmap.rightids = numpy.array(rightlist)
    idmap.start = numpy.empty((len(leftlist)), dtype="|O")
    idmap.stop = numpy.empty((len(leftlist)), dtype="|O")
    return idmap


def get_ndx_map(leftlist, rightlist):
    ndx = sidekit.Ndx()
    ndx.modelset = numpy.array(list(set(leftlist)))
    ndx.segset = numpy.array(rightlist)
    ndx.trialmask = numpy.ones((len(ndx.modelset), len(ndx.segset)), dtype='bool')
    ndx.validate()
    return ndx


def get_key_map(ndx, leftlist, rightlist):
    key = sidekit.Key()
    key.modelset = ndx.modelset
    key.segset = ndx.segset
    model_num = len(key.modelset)
    seg_num = len(key.segset)
    model_set = list(set(leftlist))
    seg_list = rightlist
    key.tar = numpy.zeros((model_num, seg_num), dtype='bool')
    for m_idx, m_name in enumerate(model_set):
        for s_idx, s_name in enumerate(seg_list):
            fm_name = s_name.split('.')[0]
            if fm_name == m_name:
                key.tar[m_idx, s_idx] = True

    key.non = numpy.ones((model_num, seg_num), dtype='bool')
    for m_idx, m_name in enumerate(model_set):
        for s_idx, s_name in enumerate(seg_list):
            fm_name = s_name.split('.')[0]
            if fm_name == m_name:
                key.non[m_idx, s_idx] = False
    key.validate()
    return key


def get_feature_extractor():
    return sidekit.FeaturesExtractor(audio_filename_structure=None,
                                     feature_filename_structure=None,
                                     sampling_frequency=None,
                                     lower_frequency=133.3,  # 200
                                     higher_frequency=4000,  # 3800
                                     filter_bank="log",
                                     filter_bank_size=40,
                                     window_size=0.025,
                                     shift=0.01,
                                     ceps_number=20,
                                     vad="snr",
                                     snr=40,
                                     pre_emphasis=0.97,
                                     save_param=["vad", "energy", "cep", "fb"],
                                     keep_all_features=True)


def get_feature_server(feat_dir):
    return sidekit.FeaturesServer(features_extractor=None,
                                  feature_filename_structure=feat_dir,
                                  sources=None,
                                  dataset_list=["energy", "cep", "vad"],
                                  mask=None,
                                  feat_norm="cmvn",
                                  global_cmvn=None,
                                  dct_pca=False,
                                  dct_pca_config=None,
                                  sdc=False,
                                  sdc_config=None,
                                  delta=True,
                                  double_delta=True,
                                  delta_filter=None,
                                  context=None,
                                  traps_dct_nb=None,
                                  rasta=True,
                                  keep_all_features=False)


def adapt_stats_with_feat_dir(feature_dir, ubm, nbThread):
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    # ubm = sidekit.Mixture(mixture_file_name=ubm_model)
    feature_filename = os.path.join(feature_dir, "{}.h5")
    features_server = get_feature_server(feature_filename)

    adapt_stat = StatServer(id_map, ubm=ubm)
    adapt_stat.accumulate_stat(
        ubm=ubm,
        feature_server=features_server,
        seg_indices=range(adapt_stat.segset.shape[0]),
        num_thread=nbThread,
        )
    return adapt_stat


def adaption_all_enrolled_speakers():
    spk_enroll_file = config.get_enroll_data
    spk_stat = adapt_stats_with_feat_dir(
        config.get_enroll_feat_dir,
        config.ubm_path
    )
    spk_i_vect, _ = extract_i_vector(spk_stat)
    if os.path.exists(spk_enroll_file):
        os.remove(spk_enroll_file)
    spk_i_vect.write(spk_enroll_file)


def extract_i_vector(stats_speaker):
    ubm = sidekit.Mixture(mixture_file_name=config.ubm_path)
    tv_file = config.tv_path
    iv, iv_uncertaniny = i_vector_with_stat(stats_speaker, ubm, tv_file)
    return iv, iv_uncertaniny


def i_vector_with_stat(stat, ubm_model, tv_file):
    tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5(tv_file)
    fa = sidekit.FactorAnalyser(mean=tv_mean, Sigma=tv_sigma, F=tv)
    iv, iv_uncertainty = fa.extract_ivectors_single(ubm=ubm_model, stat_server=stat, uncertainty=True)
    return iv, iv_uncertainty


def i_vector_with_feat_dir(feat_dir, ubm_file, tv_file):
    ubm = sidekit.Mixture(mixture_file_name=ubm_file)
    feat_stat = adapt_stats_with_feat_dir(feat_dir, ubm_file)
    return i_vector_with_stat(feat_stat, ubm, tv_file)


if __name__ == "__main__":
    feat_dir = config.get_test_feat_dir
    ubm_file = config.ubm_path
    tv_file = config.tv_path
    iv, iv_uncertainy = i_vector_with_feat_dir(feat_dir, ubm_file, tv_file)
    aa = 0
