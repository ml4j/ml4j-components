package org.ml4j.nn.components.manytoone.legacy;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.images.ChannelConcatImages;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
import org.ml4j.images.SingleChannelImages;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
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
		
		long start = new Date().getTime();
		
		boundaries = new int[gradient.size()];
		int ind = 0;
		NeuronsActivation totalActivation = null;
		/*
		int featureCount = (int)gradient.stream().mapToLong(g -> g.getFeatureCount()).sum();
		Matrix combined = context.getMatrixFactory().createMatrix(featureCount, gradient.get(0).getActivations().getColumns());
		float[] combinedData = combined.getRowByRowArray();
		int previousIndex = 0;
		for (NeuronsActivation activation : gradient) {
			int i = activation.getFeatureCount();
			
			float[] data = activation.getActivations().getRowByRowArray();
			System.arraycopy(data, 0, combinedData, previousIndex, i);
			previousIndex = previousIndex + i;
		}
		NeuronsActivation totalActivation = new NeuronsActivation(combined, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	*/
		
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
				//totalDepth = totalDepth + ((Neurons3D)activation.getNeurons()).getDepth();
				width = ((Neurons3D)activation.getNeurons()).getWidth();
				height = ((Neurons3D)activation.getNeurons()).getHeight();
			}
		
			exampleCount = activation.getExampleCount();
			featureCount = featureCount + activation.getFeatureCount();

		}
		
		//all3D = false;
		
				
		if (!all3D) {
			//if (true) throw new IllegalStateException();
			for (NeuronsActivation activation : gradient) {
				if (totalActivation == null) {

					totalActivation = activation.dup();
					boundaries[ind] = activation.getFeatureCount();
				} else {
					totalActivation.combineFeaturesInline(activation, context.getMatrixFactory());
					//activation.close();
					boundaries[ind] = activation.getFeatureCount() + boundaries[ind - 1];
				}
				ind++;
			}
			//System.out.println("Not all 3D");

			LOGGER.debug("End Combining input for many to one junction");
			
			long end = new Date().getTime();
			//Timings.addTime(TimingKey.COMBINING, end - start);
			
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
					if (true) throw new IllegalStateException();

					Neurons3D neurons3D = ((ImageNeuronsActivation)activation).getNeurons();
					Images images= null;
					if (neurons3D.getDepth() == 1) {
						images = new SingleChannelImages(activation.getActivations(context.getMatrixFactory()).getRowByRowArray(), 
								0, neurons3D.getHeight(), neurons3D.getWidth(), 0, 0, exampleCount);
					} else {
						images =  new MultiChannelImages(activation.getActivations(context.getMatrixFactory()).getRowByRowArray(), 
								neurons3D.getDepth(), neurons3D.getHeight(), neurons3D.getWidth(), 0, 0, exampleCount);
					}
					
					//Image image = imageNeuronsActivation.getImage();
					imagesList.add(images);
				}
			}
			
			Images result = new ChannelConcatImages(imagesList, height, width, 0, 0, exampleCount);

			LOGGER.debug("End Combining input for many to one junction:" + result.getChannels());
			
			long end = new Date().getTime();
			//Timings.addTime(TimingKey.COMBINING, end - start);
			
			//System.out.println("All 3D");
			//return new NeuronsActivationImpl(context.getMatrixFactory().createMatrixFromRowsByRowsArray(featureCount, exampleCount, result.getData()), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

			return new ImmutablePair<>(new ImageNeuronsActivationImpl(new Neurons3D(width, height, result.getChannels(), false), result, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false), boundaries) ;
		}
	
		
	}

	@Override
	public DefaultManyToOneFilterConcatDirectedComponentLegacy dup() {
		return new DefaultManyToOneFilterConcatDirectedComponentLegacy(pathCombinationStrategy);
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.MANY_TO_ONE;
	}

	@Override
	public ManyToOneDirectedComponentActivation forwardPropagate(List<NeuronsActivation> input,
			DirectedComponentsContext context) {
		
		Pair<NeuronsActivation, int[]> b = getCombinedOutput(input, context);
		
		return new ManyToOneFilterConcatDirectedComponentActivation(input.size(), b.getRight(), b.getLeft());
	}

	
}
