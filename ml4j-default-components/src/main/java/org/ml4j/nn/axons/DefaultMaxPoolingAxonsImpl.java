package org.ml4j.nn.axons;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

public class DefaultMaxPoolingAxonsImpl implements MaxPoolingAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private MatrixFactory matrixFactory;
	private Axons3DConfig config;

	public DefaultMaxPoolingAxonsImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config) {
		super();
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
	}

	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;

	@Override
	public Neurons3D getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons3D getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		// TODO - currently mock implementation
		return new AxonsActivationImpl(this, null, () -> leftNeuronsActivation,
				new ImageNeuronsActivationImpl(
						matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(),
								leftNeuronsActivation.getExampleCount()),
						getRightNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false),
				leftNeurons, rightNeurons);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		// TODO - currently mock implementation		
		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation,
				new ImageNeuronsActivationImpl(
						matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(),
								rightNeuronsActivation.getExampleCount()),
						getLeftNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false),
				leftNeurons, rightNeurons);
	}

	@Override
	public MaxPoolingAxons dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return false;
	}

	@Override
	public Axons3DConfig getConfig() {
		return config;
	}

}
