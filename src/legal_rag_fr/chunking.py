
def chunk_fixed(words: list[str], size: int, overlap: int) -> list[list[str]]:
    out = []
    window = size - overlap
    if not (0 <= overlap < size):
        raise ValueError("Overlap must be between 0 and size")
    for start in range(0, len(words), window):
        out.append(words[start: min(start+size, len(words))])
        if (start+size >= len(words)): break
    
    return out            

def chunk_by_paragraph(paragraphs: list[list[str]], size: int) -> list[list[str]]:
    out, cur = [], []                                                                                                                                                
    for para in paragraphs:                                                                                                                                          
        if len(cur) + len(para) <= size:                                                                                                                             
            cur += para                                                                                                             
        else:
            if cur:                                                                                                                                                        
                out.append(cur)  # flush                                                                                                                                 
            cur = para.copy()        # new accumulator starts at this para     
    if cur:                                                                                        
        out.append(cur)          # final flush — always, even if only one para                                                                                           
    return out 