package org.ml4j.nn.components;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyGenericComponent implements DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, Object> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private Neurons inputNeurons;
	private Neurons outputNeurons;
	private NeuralComponentType<? extends DefaultChainableDirectedComponent<?, ?>> neuralComponentType;

	public DummyGenericComponent(Neurons inputNeurons, Neurons outputNeurons,
			NeuralComponentType<? extends DefaultChainableDirectedComponent<?, ?>> neuralComponentType2) {
		super();
		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
		this.neuralComponentType = neuralComponentType2;
	}

	@Override
	public Object getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
		return new Object();
	}

	@Override
	public DefaultChainableDirectedComponentActivation forwardPropagate(NeuronsActivation input, Object context) {
		NeuronsActivation dummyOutput = new DummyNeuronsActivation(outputNeurons, 
				input.getFeatureOrientation(), input.getExampleCount());
		if (dummyOutput.getFeatureCount() != getOutputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException();
		}
		return new DummyGenericComponentActivation(input, dummyOutput);
	}

	@Override
	public DefaultChainableDirectedComponent<DefaultChainableDirectedComponentActivation, Object> dup() {
		return new DummyGenericComponent(inputNeurons, outputNeurons, neuralComponentType);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public Neurons getInputNeurons() {
		return inputNeurons;
	}

	@Override
	public Neurons getOutputNeurons() {
		return outputNeurons;
	}

	@Override
	public NeuralComponentType<? extends DefaultChainableDirectedComponent<?, ?>> getComponentType() {
		return neuralComponentType;
	}

}
