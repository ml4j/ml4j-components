package org.ml4j.nn.components.onetomany;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetomany.base.OneToManyDirectedComponentBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyOneToManyDirectedComponent extends OneToManyDirectedComponentBase<DummyOneToManyDirectedComponentActivation> implements OneToManyDirectedComponent<DummyOneToManyDirectedComponentActivation> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyOneToManyDirectedComponent.class);
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	@Override
	public DummyOneToManyDirectedComponentActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		LOGGER.debug("Mock splitting input neurons activation into multiple output neurons activations" );
		return new DummyOneToManyDirectedComponentActivation(neuronsActivation);
	}

	@Override
	public DummyOneToManyDirectedComponent dup() {
		return new DummyOneToManyDirectedComponent();
	}
}
