package org.ml4j.nn.components.manytoone;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.images.ChannelConcatImages;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
import org.ml4j.images.SingleChannelImages;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentBase;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultManyToOneFilterConcatDirectedComponentLegacy extends ManyToOneDirectedComponentBase<ManyToOneDirectedComponentActivation>
	implements ManyToOneDirectedComponent<ManyToOneDirectedComponentActivation>{

	public DefaultManyToOneFilterConcatDirectedComponentLegacy(PathCombinationStrategy pathCombinationStrategy) {
		super(pathCombinationStrategy);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private int[] boundaries;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultManyToOneFilterConcatDirectedComponentLegacy.class);


	protected Pair<NeuronsActivation, int[]>  getCombinedOutput(List<NeuronsActivation> gradient, DirectedComponentsContext context) {
		
		LOGGER.debug("Combining input for many to one junction");
			
		boundaries = new int[gradient.size()];
		int ind = 0;
		NeuronsActivation totalActivation = null;
		
		int featureCount = 0;
		int exampleCount = 0;
		//int totalDepth = 0;
		int width = 0;
		int height = 0;
		
		boolean all3D = true;
		for (NeuronsActivation activation : gradient) {
			if (!(activation.getNeurons() instanceof Neurons3D)) {
				all3D = false;
			} else {
				width = ((Neurons3D)activation.getNeurons()).getWidth();
				height = ((Neurons3D)activation.getNeurons()).getHeight();
			}
		
			exampleCount = activation.getExampleCount();
			featureCount = featureCount + activation.getFeatureCount();

		}		
				
		if (!all3D) {
			for (NeuronsActivation activation : gradient) {
				if (totalActivation == null) {

					totalActivation = activation.dup();
					boundaries[ind] = activation.getFeatureCount();
				} else {
					totalActivation.combineFeaturesInline(activation, context.getMatrixFactory());
					boundaries[ind] = activation.getFeatureCount() + boundaries[ind - 1];
				}
				ind++;
			}

			LOGGER.debug("End Combining input for many to one junction");
			
			return new ImmutablePair<>(new NeuronsActivationImpl(totalActivation.getActivations(context.getMatrixFactory()), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET), boundaries);
		} else {
			boolean first = true;
			for (NeuronsActivation activation : gradient) {
				if (first) {

					boundaries[ind] = activation.getFeatureCount();
					first = false;
				} else {
					//activation.close();
					boundaries[ind] = activation.getFeatureCount() + boundaries[ind - 1];
				}
				ind++;
			}
			List<Images> imagesList = new ArrayList<>();
			for (NeuronsActivation activation : gradient) {
				if (activation instanceof ImageNeuronsActivation) {
					imagesList.add(((ImageNeuronsActivation)activation).getImages());
				} else {
					Neurons3D neurons3D = ((ImageNeuronsActivation)activation).getNeurons();
					Images images= null;
					if (neurons3D.getDepth() == 1) {
						images = new SingleChannelImages(activation.getActivations(context.getMatrixFactory()).getRowByRowArray(), 
								0, neurons3D.getHeight(), neurons3D.getWidth(), 0, 0, exampleCount);
					} else {
						images =  new MultiChannelImages(activation.getActivations(context.getMatrixFactory()).getRowByRowArray(), 
								neurons3D.getDepth(), neurons3D.getHeight(), neurons3D.getWidth(), 0, 0, exampleCount);
					}
					
					imagesList.add(images);
				}
			}
			
			Images result = new ChannelConcatImages(imagesList, height, width, 0, 0, exampleCount);

			LOGGER.debug("End Combining input for many to one junction:" + result.getChannels());
			
			return new ImmutablePair<>(new ImageNeuronsActivationImpl(new Neurons3D(width, height, result.getChannels(), false), result, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false), boundaries) ;
		}
	
		
	}

	@Override
	public DefaultManyToOneFilterConcatDirectedComponentLegacy dup() {
		return new DefaultManyToOneFilterConcatDirectedComponentLegacy(pathCombinationStrategy);
	}

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.getBaseType(NeuralComponentBaseType.MANY_TO_ONE);
	}

	@Override
	public ManyToOneDirectedComponentActivation forwardPropagate(List<NeuronsActivation> input,
			DirectedComponentsContext context) {
		
		Pair<NeuronsActivation, int[]> b = getCombinedOutput(input, context);
		
		return new  ManyToOneFilterConcatDirectedComponentActivation(input.size(), b.getRight(), b.getLeft());
	}

	
}
