package org.ml4j.nn.components.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentBase;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyBatchNormDirectedAxonsComponent<L extends Neurons, R extends Neurons> extends DirectedAxonsComponentBase<L, R, Axons<? extends L, ? extends R, ?>> 
	implements BatchNormDirectedAxonsComponent<L, R> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDirectedAxonsComponent.class);
	/**
	 * Defaut serialization id;
	 */
	private static final long serialVersionUID = 1L;

	
	public DummyBatchNormDirectedAxonsComponent(Axons<? extends L, ? extends R, ?> axons) {
		super(axons);
	}

	@Override
	public float getBetaForExponentiallyWeightedAverages() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureMeans(Matrix arg0) {
		throw new UnsupportedOperationException();		
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureVariances(Matrix arg0) {
		throw new UnsupportedOperationException();		
	}

	@Override
	public BatchNormDirectedAxonsComponent<L, R> dup() {
		return new DummyBatchNormDirectedAxonsComponent<>(axons.dup());
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation neuronsActivation, AxonsContext axonsContext) {
		LOGGER.debug("Forward propagating through DummyDirectedAxonsComponent");
		return new DummyDirectedAxonsComponentActivation(this, new DummyNeuronsActivation(axons.getRightNeurons(), 
				neuronsActivation.getFeatureOrientation(), neuronsActivation.getExampleCount()));
	}

}
