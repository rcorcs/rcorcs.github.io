/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos.
 */
import java.util.List;

public abstract class Busca {
	private Digrafo g;
	private int verticeOrigem;
	
	public Busca(Digrafo g, int verticeOrigem){
		this.g = g;
		this.verticeOrigem = verticeOrigem;
		realizaBusca();
	}
	
	public Digrafo digrafo(){
		return this.g;
	}
	
	public int verticeOrigem(){
		return this.verticeOrigem;
	}
	
	protected abstract void realizaBusca();	
	
	public abstract boolean existeCaminhoPara(int verticeDestino);
	
	public abstract List<Integer> caminhoPara(int verticeDestino);

	public abstract double pesoCaminhoPara(int verticeDestino);
}
