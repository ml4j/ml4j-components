package org.ml4j.nn.components.onetoone;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import org.ml4j.Matrix;
import org.ml4j.images.Images;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;

public class DefaultSpaceToDepthDirectedComponent implements DefaultChainableDirectedComponent<DefaultSpaceToDepthDirectedComponentActivation, DirectedComponentsContext> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private int blockHeight;
	private int blockWidth;
	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;
	protected String name;

	public DefaultSpaceToDepthDirectedComponent(String name, Neurons3D leftNeurons, Neurons3D rightNeurons, int blockHeight, int blockWidth) {
		this.blockHeight = blockHeight;
		this.blockWidth = blockWidth;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.name = name;
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext) {
		return directedComponentsContext;
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return format.equals(ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT);
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public DefaultChainableDirectedComponent<DefaultSpaceToDepthDirectedComponentActivation, DirectedComponentsContext> dup() {
		return new DefaultSpaceToDepthDirectedComponent(name, leftNeurons, rightNeurons, blockHeight, blockWidth);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public Neurons getInputNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons getOutputNeurons() {
		return rightNeurons;
	}

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.createSubType(NeuralComponentBaseType.CUSTOM, "SPACE_TO_DEPTH");
	}

	@Override
	public DefaultSpaceToDepthDirectedComponentActivation forwardPropagate(NeuronsActivation input,
			DirectedComponentsContext context) {
		Images images = input.asImageNeuronsActivation(leftNeurons, DimensionScope.INPUT).getImages();
		Matrix output = images.spaceToDepthExport(context.getMatrixFactory(), blockHeight, blockWidth);
		ImageNeuronsActivation activation =  new ImageNeuronsActivationImpl(output, 
				rightNeurons, ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);
		return new DefaultSpaceToDepthDirectedComponentActivation(this, activation);
	}

	@Override
	public String getName() {
		return name;
	}

}
