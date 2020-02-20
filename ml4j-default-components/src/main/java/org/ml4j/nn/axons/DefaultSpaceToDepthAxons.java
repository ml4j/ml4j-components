package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.Matrix;
import org.ml4j.images.Images;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;

public class DefaultSpaceToDepthAxons extends AxonsBase<Neurons3D, Neurons3D, DefaultSpaceToDepthAxons, AxonsConfig<Neurons3D, Neurons3D>> {

	public static final AxonsType SPACE_TO_DEPTH_AXONS_TYPE = AxonsType.createCustomBaseType("SPACE_TO_DEPTH");
	public static final Class<DefaultSpaceToDepthAxons> SPACE_TO_DEPTH_AXONS_CLASS = DefaultSpaceToDepthAxons.class;

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private int blockHeight;
	private int blockWidth;

	public DefaultSpaceToDepthAxons(AxonsConfig<Neurons3D, Neurons3D> axonsConfig) {
		super(axonsConfig);
		this.blockHeight = axonsConfig.getLeftNeurons().getHeight() / axonsConfig.getRightNeurons().getHeight();
		this.blockWidth = axonsConfig.getLeftNeurons().getWidth() / axonsConfig.getRightNeurons().getWidth();
	}
	
	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation input, AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		Images images = input.asImageNeuronsActivation(getLeftNeurons(), DimensionScope.INPUT).getImages();
		Matrix output = images.spaceToDepthExport(axonsContext.getMatrixFactory(), blockHeight, blockWidth);
		ImageNeuronsActivation outputActivation =  new ImageNeuronsActivationImpl(output, 
				getRightNeurons(), ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);
		return new AxonsActivationImpl(this, null, () -> input, outputActivation);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation input, AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		// TODO
		throw new UnsupportedOperationException("Not yet implemented");
	}
	
	@Override
	public AxonsType getAxonsType() {
		return SPACE_TO_DEPTH_AXONS_TYPE;
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return false;
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> neuronsActivationFormat) {
		return neuronsActivationFormat.equals(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT);
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public DefaultSpaceToDepthAxons dup() {
		return new DefaultSpaceToDepthAxons(axonsConfig.dup());
	}
}
