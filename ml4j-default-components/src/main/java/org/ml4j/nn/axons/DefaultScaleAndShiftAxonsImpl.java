package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.Neurons;

public class DefaultScaleAndShiftAxonsImpl<N extends Neurons> extends WeightedAxonsBase<N, N, ScaleAndShiftAxons<N>>
		implements ScaleAndShiftAxons<N> {

	public DefaultScaleAndShiftAxonsImpl(N leftNeurons, N rightNeurons, AxonWeights axonWeights) {
		super(leftNeurons, rightNeurons, axonWeights);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public ScaleAndShiftAxons<N> dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return !axonsContext.isWithFreezeOut();
	}

	@Override
	public Matrix getScaleColumnVector() {
		return axonWeights.getConnectionWeights();
	}

	@Override
	public Matrix getShiftColumnVector() {
		return axonWeights.getLeftToRightBiases();
	}

	@Override
	protected boolean isLeftInputDropoutSupported() {
		return false;
	}
}
