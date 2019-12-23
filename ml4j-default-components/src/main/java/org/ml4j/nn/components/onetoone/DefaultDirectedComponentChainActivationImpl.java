package org.ml4j.nn.components.onetoone;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainActivationImpl extends DefaultDirectedComponentChainActivationBase<DefaultDirectedComponentChain>
		implements DefaultDirectedComponentChainActivation {
	
	private List<DefaultChainableDirectedComponentActivation> activations;

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return activations.stream().flatMap(a -> a.decompose().stream()).collect(Collectors.toList());
	}

	public DefaultDirectedComponentChainActivationImpl(DefaultDirectedComponentChain componentChain, List<DefaultChainableDirectedComponentActivation> activations,
			NeuronsActivation output) {
		super(componentChain, output);
		this.activations = activations;
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> getActivations() {
		return activations;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		List<DefaultChainableDirectedComponentActivation> reversedSynapseActivations =
		        new ArrayList<>();
		    reversedSynapseActivations.addAll(getActivations());
		    Collections.reverse(reversedSynapseActivations);
		    return backPropagateAndAddToSynapseGradientList(outerGradient,
		        reversedSynapseActivations);
	}
	
	private DirectedComponentGradient<NeuronsActivation> backPropagateAndAddToSynapseGradientList(
		      DirectedComponentGradient<NeuronsActivation> outerSynapsesGradient,
		      List<DefaultChainableDirectedComponentActivation> activationsToBackPropagateThrough) {

			List<Supplier<AxonsGradient>> totalTrainableAxonsGradients = new ArrayList<>();
			totalTrainableAxonsGradients.addAll(outerSynapsesGradient.getTotalTrainableAxonsGradients());
			
		    DirectedComponentGradient<NeuronsActivation> finalGrad = outerSynapsesGradient;
		    DirectedComponentGradient<NeuronsActivation> synapsesGradient = outerSynapsesGradient;
		    List<Supplier<AxonsGradient>> finalTotalTrainableAxonsGradients = outerSynapsesGradient.getTotalTrainableAxonsGradients();
		    List<DirectedComponentGradient<NeuronsActivation>> componentGradients = new ArrayList<>();
		    for (ChainableDirectedComponentActivation<NeuronsActivation> synapsesActivation : activationsToBackPropagateThrough) {
		     
		      componentGradients.add(synapsesGradient);
		      synapsesGradient = 
		          synapsesActivation.backPropagate(synapsesGradient);
		   
		      finalTotalTrainableAxonsGradients = synapsesGradient.getTotalTrainableAxonsGradients();
		      finalGrad = synapsesGradient;
		    }
		    return new DirectedComponentGradientImpl<>(finalTotalTrainableAxonsGradients, finalGrad.getOutput());
		  }
}
