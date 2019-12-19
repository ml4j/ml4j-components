package org.ml4j.nn.components.onetomany;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetomany.base.OneToManyDirectedComponentBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mock implementation of a  OneToManyDirectedComponent - a directed component which takes a single NeuronsActivation instance as input 
 * and map to many NeuronsActivation instances as output.
 * 
 * Used within component graphs where the flow through the NeuralNetwork is split into paths, eg. for skip-connections in ResNets or inception modules.
 * 
 * @author Michael Lavelle
 *
 * @param <A> The type of activation produced by this component on forward-propagation.
 */
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
