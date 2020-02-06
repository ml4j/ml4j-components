package org.ml4j.nn.components.manytoone;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.ml4j.images.Images;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ManyToOneFilterConcatDirectedComponentActivation extends DefaultManyToOneDirectedComponentActivationImpl {

	private static final Logger LOGGER = LoggerFactory
			.getLogger(ManyToOneFilterConcatDirectedComponentActivation.class);

	private int[] channelBoundaries;
	private Neurons3D outputNeurons;

	public ManyToOneFilterConcatDirectedComponentActivation(int size, int[] channelBoundaries,
			ImageNeuronsActivation output) {
		super(size, output);
		this.outputNeurons = output.getNeurons();
		this.channelBoundaries = channelBoundaries;
		if (channelBoundaries == null) {
			throw new IllegalArgumentException();
		}
	}

	private Images getChannelImages(Images images, int channelBoundaryIndex) {
		return channelBoundaryIndex == 0 ? images.getChannels(0, channelBoundaries[channelBoundaryIndex])
				: images.getChannels(channelBoundaries[channelBoundaryIndex - 1],
						channelBoundaries[channelBoundaryIndex]);
	}

	@Override
	public DirectedComponentGradient<List<NeuronsActivation>> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {

		LOGGER.debug("Splitting gradient for many to one filter");

		// Convert the outer gradient into image format
		ImageNeuronsActivation imagesActivation = outerGradient.getOutput().asImageNeuronsActivation(outputNeurons, DimensionScope.OUTPUT);
		Images outerGradientAsImage = imagesActivation.getImages();

		// Split according to the filter channel boundaries into multiple
		// back-propagated gradients activations.
		List<NeuronsActivation> outerGradientBackPropagatedImages = IntStream.range(0, channelBoundaries.length)
				.mapToObj(i -> getChannelImages(outerGradientAsImage, i))
				.map(channelImage -> new ImageNeuronsActivationImpl(
						new Neurons3D(outputNeurons.getWidth(), outputNeurons.getHeight(), channelImage.getChannels(),
								false),
						channelImage, imagesActivation.getFormat(), imagesActivation.isImmutable()))
				.collect(Collectors.toList());

		outerGradientBackPropagatedImages.stream().forEach(i -> i.setImmutable(true));

		LOGGER.debug("End splitting gradient for many to one filter");

		return new DirectedComponentGradientImpl<>(outerGradient.getTotalTrainableAxonsGradients(),
				outerGradientBackPropagatedImages);
	}
}
