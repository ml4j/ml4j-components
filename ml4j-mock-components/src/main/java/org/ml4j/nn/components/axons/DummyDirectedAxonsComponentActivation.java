package org.ml4j.nn.components.axons;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyDirectedAxonsComponentActivation extends DirectedAxonsComponentActivationBase implements DirectedAxonsComponentActivation {
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDirectedAxonsComponentActivation.class);

	public DummyDirectedAxonsComponentActivation(DirectedAxonsComponent<?, ?> axonsComponent, NeuronsActivation output) {
		super(axonsComponent, output);
	}
	
	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		LOGGER.debug("Back propagating gradient through DummyDirectedAxonsComponentActivation");
		return gradient;
	}

	@Override
	public float getTotalRegularisationCost() {
		return 0;
	}
}
