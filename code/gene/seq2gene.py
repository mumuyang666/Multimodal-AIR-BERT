from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

def run_blast(query_sequence):
    # 使用NCBI BLAST进行在线比对
    result_handle = NCBIWWW.qblast("blastp", "refseq_protein", query_sequence)
    
    # 解析BLAST结果
    blast_records = NCBIXML.parse(result_handle)
    
    # 提取比对结果中的V、D和J基因信息以及连接区域信息
    vdj_info = extract_vdj_info(blast_records)
    
    return vdj_info

def extract_vdj_info(blast_records):
    vdj_info = {"V": None, "D": None, "J": None, "junction": None}

    # 遍历BLAST比对结果
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            # 根据标题判断基因类型（V、D或J）
            gene_type = None
            if "V_gene" in alignment.title:
                gene_type = "V"
            elif "D_gene" in alignment.title:
                gene_type = "D"
            elif "J_gene" in alignment.title:
                gene_type = "J"
            
            # 提取V、D和J基因的位置信息
            if gene_type and not vdj_info[gene_type]:
                for hsp in alignment.hsps:
                    vdj_info[gene_type] = (hsp.query_start, hsp.query_end)
    
    # 根据V、D和J基因的位置信息提取连接区域信息
    if vdj_info["V"] and vdj_info["J"]:
        vdj_info["junction"] = (vdj_info["V"][1], vdj_info["J"][0])

    return vdj_info

# 示例：使用模拟的TCR序列进行BLAST比对
query_sequence = "CASSLPPRSGAAYNEQFF"
vdj_info = run_blast(query_sequence)

print("V gene:", vdj_info["V"])
print("D gene:", vdj_info["D"])
print("J gene:", vdj_info["J"])
print("Junction region:", vdj_info["junction"])