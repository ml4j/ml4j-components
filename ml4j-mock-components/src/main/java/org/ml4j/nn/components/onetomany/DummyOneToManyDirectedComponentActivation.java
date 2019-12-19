package org.ml4j.nn.components.onetomany;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.onetomany.base.OneToManyDirectedComponentActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyOneToManyDirectedComponentActivation extends OneToManyDirectedComponentActivationBase implements OneToManyDirectedComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyOneToManyDirectedComponentActivation.class);
	
	public DummyOneToManyDirectedComponentActivation(NeuronsActivation input) {
		super(input);
	}
	
	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<List<NeuronsActivation>> gradient) {
		LOGGER.debug("Mock back propagating multiple gradient neurons activations into a single combined neurons activation");
		return new DirectedComponentGradientImpl<>(gradient.getOutput().get(0));
	}
}
