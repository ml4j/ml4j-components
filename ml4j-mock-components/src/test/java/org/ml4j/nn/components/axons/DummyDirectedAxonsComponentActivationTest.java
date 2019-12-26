package org.ml4j.nn.components.axons;

import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.neurons.Neurons;


import org.ml4j.nn.components.axons.base.DirectedAxonsComponentActivationTestBase;

public class DummyDirectedAxonsComponentActivationTest extends DirectedAxonsComponentActivationTestBase {

	@Override
	protected <L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> DirectedAxonsComponentActivation createDirectedAxonsComponentActivationUnderTest(DirectedAxonsComponent<L, R, A> axonsComponent, AxonsActivation axonsActivation, AxonsContext axonsContext) {
		return new DummyDirectedAxonsComponentActivation<>(axonsComponent, axonsActivation, axonsContext);
	}

}
