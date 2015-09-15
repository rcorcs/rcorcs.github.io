/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos.
 */
import java.util.List;

public abstract class BuscaProfundidade extends Busca {
	private Digrafo g;
	private int verticeOrigem;
	
	public BuscaProfundidade(Digrafo g, int verticeOrigem){
		super(g,verticeOrigem);
	}
	
	protected void realizaBusca(){
		//FAZER
	}	
	
	public boolean existeCaminhoPara(int verticeDestino){
		//FAZER
		return false;
	}
	
	public List<Integer> caminhoPara(int verticeDestino){
		//FAZER
		return null;
	}
	
	public double pesoCaminhoPara(int verticeDestino){
		//FAZER
		return null;
	}
}
