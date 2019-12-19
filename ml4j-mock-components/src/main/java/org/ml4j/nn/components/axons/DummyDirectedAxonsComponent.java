package org.ml4j.nn.components.axons;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentBase;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyDirectedAxonsComponent<L extends Neurons, R extends Neurons> extends DirectedAxonsComponentBase<L, R> implements DirectedAxonsComponent<L, R> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDirectedAxonsComponent.class);

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	public DummyDirectedAxonsComponent(Axons<? extends L, ? extends R, ?> axons) {
		super(axons);
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation neuronsActivation, AxonsContext context) {
		LOGGER.debug("Forward propagating through DummyDirectedAxonsComponent");
		return new DummyDirectedAxonsComponentActivation(this, new DummyNeuronsActivation(axons.getRightNeurons(), 
				neuronsActivation.getFeatureOrientation(), neuronsActivation.getExampleCount()));
	}

	@Override
	public DirectedAxonsComponent<L, R> dup() {
		return new DummyDirectedAxonsComponent<>(axons.dup());
	}
	
}
