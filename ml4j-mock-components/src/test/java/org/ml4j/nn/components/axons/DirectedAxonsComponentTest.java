package org.ml4j.nn.components.axons;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentTestBase;
import org.ml4j.nn.neurons.Neurons;

public class DirectedAxonsComponentTest extends DirectedAxonsComponentTestBase {

	@Override
	protected <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R> createDirectedAxonsComponentUnderTest(
			Axons<L, R, ?> axons) {
		return new DummyDirectedAxonsComponent<>(axons);
	}
}
