import h5py
import numpy as np
# %matplotlib inline
import modisco
import IntegratedGradients
import modisco.visualization
from modisco.visualization import viz_sequence

# this part shows visualization of the data
# viz_sequence.plot_weights(IntegratedGradients.calc_IG()[1]['task'][0], subticks_frequency=20)
# viz_sequence.plot_weights(IntegratedGradients.calc_IG()[0][0], subticks_frequency=20)


# the following part calculates tf_modisco results

#Uncomment to refresh modules for when tweaking code during development:
from importlib import reload
reload(modisco.util)
reload(modisco.pattern_filterer)
reload(modisco.aggregator)
reload(modisco.core)
reload(modisco.seqlet_embedding.advanced_gapped_kmer)
reload(modisco.affinitymat.transformers)
reload(modisco.affinitymat.core)
reload(modisco.affinitymat)
reload(modisco.cluster.core)
reload(modisco.cluster)
reload(modisco.tfmodisco_workflow.seqlets_to_patterns)
reload(modisco.tfmodisco_workflow)
reload(modisco)


null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                    #Slight modifications from the default settings
                    sliding_window_size=15,
                    flank_size=5,
                    target_seqlet_fdr=0.15,
                    seqlets_to_patterns_factory=
                     modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                        #Note: as of version 0.5.6.0, it's possible to use the results of a motif discovery
                        # software like MEME to improve the TF-MoDISco clustering. To use the meme-based
                        # initialization, you would specify the initclusterer_factory as shown in the
                        # commented-out code below:
                        #initclusterer_factory=modisco.clusterinit.memeinit.MemeInitClustererFactory(
                        #    meme_command="meme", base_outdir="meme_out",
                        #    max_num_seqlets_to_use=10000, nmotifs=10, n_jobs=1),
                        trim_to_window_size=15,
                        initial_flank_to_add=5,
                        final_flank_to_add=5,
                        final_min_cluster_size=60,
                        #use_pynnd=True can be used for faster nn comp at coarse grained step
                        # (it will use pynndescent), but note that pynndescent may crash
                        #use_pynnd=True,
                        n_cores=10)
                )(
                 task_names=['task'],
                 contrib_scores=IntegratedGradients.calc_IG_mult_samples()[1],
                 hypothetical_contribs=IntegratedGradients.calc_IG_mult_samples()[1],
                 one_hot=IntegratedGradients.calc_IG_mult_samples()[0],
                 null_per_pos_scores=null_per_pos_scores)

