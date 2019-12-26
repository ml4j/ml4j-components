package org.ml4j.nn.components.onetoone;

import java.util.List;

import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainActivationTestBase;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDirectedComponentChainActivationTest extends DefaultDirectedComponentChainActivationTestBase {

	@Override
	protected DefaultDirectedComponentChainActivation createDefaultDirectedComponentChainActivationUnderTest(
			DefaultDirectedComponentChain componentChain, List<DefaultChainableDirectedComponentActivation> activations, NeuronsActivation output) {
		return new DummyDefaultDirectedComponentChainActivation(componentChain, activations, output);
	}

}
