package org.ml4j.nn.components.manytoone;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyManyToOneDirectedComponent extends ManyToOneDirectedComponentBase<DummyManyToOneDirectedComponentActivation> implements ManyToOneDirectedComponent<DummyManyToOneDirectedComponentActivation> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyManyToOneDirectedComponent.class);
	
	/**s
	 * Serialization id.
	 */
	private static final long serialVersionUID = -7049642040068320620L;

	@Override
	public DummyManyToOneDirectedComponentActivation forwardPropagate(List<NeuronsActivation> neuronActivations,
			DirectedComponentsContext context) {
		LOGGER.debug("Mock combining multiple neurons activations into a single output neurons activation") ;
		return new DummyManyToOneDirectedComponentActivation(neuronActivations.size(), neuronActivations.get(0));
	}

	@Override
	public ManyToOneDirectedComponent<DummyManyToOneDirectedComponentActivation> dup() {
		return new DummyManyToOneDirectedComponent();
	}
}
