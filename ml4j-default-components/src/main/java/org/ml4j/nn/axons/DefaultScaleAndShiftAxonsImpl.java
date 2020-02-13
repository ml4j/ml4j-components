package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.Neurons;

public class DefaultScaleAndShiftAxonsImpl<N extends Neurons> extends WeightedAxonsBase<N, N, ScaleAndShiftAxons<N>, AxonsConfig<N, N>>
		implements ScaleAndShiftAxons<N> {

	public DefaultScaleAndShiftAxonsImpl(AxonsConfig<N, N> axonsConfig, AxonWeights axonWeights) {
		super(axonsConfig, axonWeights);
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
		return axonWeights.getConnectionWeights().getWeights();
	}

	@Override
	public Matrix getShiftColumnVector() {
		return axonWeights.getLeftToRightBiases().getWeights();
	}

	@Override
	protected boolean isLeftInputDropoutSupported() {
		return false;
	}
	
	@Override
	public AxonsType getAxonsType() {
		return AxonsType.getBaseType(AxonsBaseType.SCALE_AND_SHIFT);
	}
}
