import sys
sys.path.append('./')
from torch.utils.data import DataLoader
from evaluation.Eval_Rouge import *
from evaluation.Eval_Bleu import *
from evaluation.Eval_Meteor import *
from evaluation.Eval_F1 import *
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from TAKE.Utils import *
import json
import os
import logging
from datetime import datetime

# 即时刷新的文件处理器
class FlushFileHandler(logging.FileHandler):
    """每次写入后立即刷新的文件处理器"""
    def emit(self, record):
        super().emit(record)
        self.flush()

# 配置日志格式
def setup_logger(name, log_file=None):
    """设置带时间戳的日志"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除已有的 handlers
    logger.handlers = []

    # 控制台输出（即时刷新）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（即时刷新）
    if log_file:
        file_handler = FlushFileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def rounder(num, places):
    return round(num, places)

def train_embedding(model):
    for name, param in model.named_parameters():
        if 'embedding' in name:
            param.requires_grad = True


def init_params(model, escape=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if escape is not None and escape in name:
            continue
        if param.data.dim() > 1:
            xavier_normal_(param.data)


def freeze_params(model, freeze=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if freeze is not None and freeze in name:
            param.requires_grad = False


def unfreeze_params(model, unfreeze=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if unfreeze is not None and unfreeze in name:
            param.requires_grad = True

class CumulativeTrainer(object):
    def __init__(self, name, model, tokenizer, detokenizer, local_rank=None, accumulation_steps=None, log_dir=None):
        super(CumulativeTrainer, self).__init__()
        self.local_rank = local_rank
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.name = name

        # 设置日志
        log_file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'train_{timestamp}.log')
        self.logger = setup_logger(f'trainer_{name}', log_file)
        self.logger.info(f'=== Training session started: {name} ===')
        if log_file:
            self.logger.info(f'Log file: {log_file}')

        if torch.cuda.is_available():
            self.model = model.cuda()
            self.logger.info('Using CUDA')
        else:
            self.model = model
            self.logger.info('Using CPU')

        self.eval_model = self.model
        self.accumulation_steps = accumulation_steps
        self.accumulation_count = 0

#train mini-batch
    def train_batch(self, epoch, data, method, optimizer, scheduler=None):
        self.accumulation_count += 1

        loss_ks, loss_distill, final_ks_acc, shift_ks_acc, inherit_ks_acc, s_shift_prob, loss_ID, ID_acc = self.model(data, method=method, epoch=epoch)
        lambda_weight = 0.5
        loss_primary = loss_ks
        loss_aboutID = loss_distill + loss_ID
        loss = (loss_primary + loss_aboutID * lambda_weight) / self.accumulation_steps

        loss.backward()

        if self.accumulation_count % self.accumulation_steps == 0:
            # The norm is computed over all gradients together,
            # as if they were concatenated into a single vector.
            # return is Total norm of the parameters (viewed as a single vector).

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)

            # torch.optim.Adam.step()
            # Performs a single optimization step.
            optimizer.step()
            if scheduler is not None:
                # Learning rate scheduling should be applied after optimizer’s update
                scheduler.step()
            optimizer.zero_grad()



        return loss_ks.cpu().item(), loss_distill.cpu().item(), \
               final_ks_acc, shift_ks_acc, inherit_ks_acc, \
               s_shift_prob.cpu().item(), loss_ID.cpu().item(), ID_acc

    def serialize(self, epoch, scheduler, saved_model_path):

        fuse_dict = {"model": self.eval_model.state_dict(), "scheduler": scheduler.state_dict()}

        torch.save(fuse_dict, os.path.join(saved_model_path, '.'.join([str(epoch), 'pkl'])))
        self.logger.info(f"Model saved: {os.path.join(saved_model_path, str(epoch) + '.pkl')}")

        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)

        checkpoints["time"].append(epoch)

        with open(saved_model_path + "checkpoints.json", 'w', encoding='utf-8') as w:
            json.dump(checkpoints, w)


    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer, scheduler=None):
        self.model.train()  # Sets the module in training mode；
        if torch.cuda.is_available():
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)

        total_batches = len(train_loader)
        self.logger.info(f'{"="*60}')
        self.logger.info(f'Starting Epoch {epoch} | Total batches: {total_batches} | Batch size: {batch_size}')
        self.logger.info(f'{"="*60}')

        start_time = time.time()
        count_batch = 0

        accu_loss_ks = 0.
        accu_loss_distill = 0.   
        accu_loss_ID = 0.
        accu_final_ks_acc = 0.
        accu_shift_ks_acc = 0.
        accu_inherit_ks_acc = 0.
        accu_ID_acc = 0.
        accu_s_shift_prob = 0.

        step = 0

        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda
            

            count_batch += 1

            loss_ks, loss_distill, final_ks_acc, shift_ks_acc, inherit_ks_acc, s_shift_prob, loss_ID, ID_acc = self.train_batch(epoch, data, method=method, optimizer=optimizer, scheduler=scheduler)
            #accumulate loss
            accu_loss_ks += loss_ks
            accu_loss_distill += loss_distill
            accu_loss_ID += loss_ID
            accu_final_ks_acc += final_ks_acc
            accu_shift_ks_acc += shift_ks_acc
            accu_inherit_ks_acc += inherit_ks_acc
            accu_ID_acc += ID_acc
            accu_s_shift_prob += s_shift_prob
            step +=1

            # 每 10 个 batch 输出一次日志（可调整）
            log_interval = 10
            if j >= 0 and j % log_interval == 0:
                elapsed_time = time.time() - start_time
                total_batches = len(train_loader)
                progress_pct = (j + 1) / total_batches * 100

                if scheduler is not None:
                    lr_str = f"{scheduler.get_last_lr()[0]:.2e}" if scheduler.get_last_lr() else "N/A"
                else:
                    lr_str = "N/A"

                self.logger.info(
                    f'[Epoch {epoch}] Batch {j+1}/{total_batches} ({progress_pct:.1f}%) | '
                    f'loss_ks: {rounder(accu_loss_ks / max(step, 1), 4):.4f} | '
                    f'loss_distill: {rounder(accu_loss_distill / max(step, 1), 4):.4f} | '
                    f'loss_ID: {rounder(accu_loss_ID / max(step, 1), 4):.4f} | '
                    f'ks_acc: {rounder(accu_final_ks_acc / max(step, 1), 4):.4f} | '
                    f'ID_acc: {rounder(accu_ID_acc / max(step, 1), 4):.4f} | '
                    f'elapsed: {rounder(elapsed_time, 1)}s | LR: {lr_str}'
                )

                accu_loss_ks = 0.
                accu_loss_distill = 0.
                accu_loss_ID = 0.
                accu_final_ks_acc = 0.
                accu_shift_ks_acc = 0.
                accu_inherit_ks_acc = 0.
                accu_ID_acc = 0.
                accu_s_shift_prob = 0.

                step = 0

                sys.stdout.flush()

            del loss_ks
            del loss_distill
            del loss_ID
            del final_ks_acc
            del shift_ks_acc
            del inherit_ks_acc
            del ID_acc
            del s_shift_prob


        # Epoch 完成日志
        epoch_time = time.time() - start_time
        self.logger.info(f'{"="*60}')
        self.logger.info(f'Epoch {epoch} completed | Total time: {epoch_time:.1f}s | Batches: {count_batch}')
        self.logger.info(f'{"="*60}')
        sys.stdout.flush()


    def predict(self, method, dataset, collate_fn, batch_size, epoch, output_path):
        #  changes the forward() behaviour of the module it is called upon. eg, it disables dropout and has batch norm use the entire population statistics
        self.eval_model.eval()   

        with torch.no_grad():
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

            accumulative_final_ks_pred = []
            accumulative_shift_ks_pred = []
            accumulative_inherit_ks_pred = []
            accumulative_knowledge_label = []
            accumulative_ID_pred = []
            accumulative_ID_label = []
            accumulative_episode_mask = []
            shift_top3_records = []
            shift_pred_records = []
            episode_offset = 0

            for k, data in enumerate(test_loader, 0):
                print("doing {} / total {} in {}".format(k+1, len(test_loader), epoch))
                
                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda
             
                # final_ks_pred [batch * max_episode_length]
                final_ks_pred, shift_ks_pred, inherit_ks_pred, ID_pred = self.eval_model(data, method=method)  #此处数据是按批传入的

                accumulative_ID_pred.append(ID_pred) # [[batch * max_episode_length],...]
                accumulative_final_ks_pred.append(final_ks_pred)  # [[batch * max_episode_length],...]
                accumulative_shift_ks_pred.append(shift_ks_pred)
                accumulative_inherit_ks_pred.append(inherit_ks_pred)
                accumulative_ID_label.append(data['Initiative_label'].reshape(-1)) # [[batch * max_episode_length],...]
                accumulative_knowledge_label.append(data['knowledge_label'].reshape(-1)) # [[batch * max_episode_length],...]
                accumulative_episode_mask.append(data['episode_mask'].reshape(-1))  # [[batch * max_episode_length],...]

                batch_size, max_episode_length = data['episode_mask'].size()

                id_pred_batch = ID_pred.reshape(batch_size, max_episode_length).cpu().tolist()
                mask_batch = data['episode_mask'].cpu().tolist()
                node_ids_batch = data.get('node_ids', torch.full_like(data['episode_mask'], -1)).cpu().tolist()

                centrality_loader = getattr(self.eval_model, "centrality_loader", None)
                for b in range(batch_size):
                    episode_index = episode_offset + b
                    if episode_index >= len(dataset.episodes):
                        continue
                    episode_examples = dataset.episodes[episode_index]
                    dialog_id = episode_examples[0]['query_id'].rsplit("_", 1)[0] if episode_examples else f"episode_{episode_index}"

                    # 6.3：逐句 shift 0/1 預測標籤（含 turn_id 以便回溯到 tiage_anno_nodes_all.csv）
                    for t in range(max_episode_length):
                        if not mask_batch[b][t]:
                            continue
                        if t >= len(episode_examples):
                            continue
                        ex = episode_examples[t]
                        query_id = ex.get("query_id", "")
                        node_id = int(node_ids_batch[b][t])
                        pred_shift = int(id_pred_batch[b][t])

                        turn_id = -1
                        if centrality_loader is not None and node_id != -1 and hasattr(centrality_loader, "get_turn_id"):
                            try:
                                turn_id = int(centrality_loader.get_turn_id(node_id))
                            except Exception:
                                turn_id = -1
                        if turn_id == -1:
                            # fallback：episode 內的 index
                            turn_id = t

                        shift_pred_records.append({
                            "dialog_id": dialog_id,
                            "query_id": query_id,
                            "turn_id": turn_id,
                            "node_id": node_id,
                            "pred_shift": pred_shift
                        })
                    shift_indices = [
                        t for t in range(max_episode_length)
                        if mask_batch[b][t] and id_pred_batch[b][t] == 1
                    ]
                    shift_found = len(shift_indices) > 0
                    # 6.1 / 6.2：以「區間 Top-3」定義輸出，並加入 turn_id 供回溯
                    shift_events = []
                    prev_shift_t = -1
                    # 區間規則（固定一條可執行規則避免重疊）：
                    # - 第一次 shift：區間 [0, cur_t]（包含本次 shift）
                    # - 後續 shift：區間 [prev_shift_t + 1, cur_t]（不含前一次 shift，包含本次 shift）
                    for cur_t in shift_indices:
                        if cur_t >= len(episode_examples):
                            continue
                        interval_start = 0 if prev_shift_t < 0 else (prev_shift_t + 1)
                        interval_end = cur_t

                        # shift 句子資訊
                        cur_ex = episode_examples[cur_t]
                        cur_query_id = cur_ex.get("query_id", "")
                        cur_node_id = int(node_ids_batch[b][cur_t])

                        cur_turn_id = -1
                        if centrality_loader is not None and cur_node_id != -1 and hasattr(centrality_loader, "get_turn_id"):
                            try:
                                cur_turn_id = int(centrality_loader.get_turn_id(cur_node_id))
                            except Exception:
                                cur_turn_id = -1
                        if cur_turn_id == -1:
                            cur_turn_id = cur_t

                        cur_tokens = dataset.query.get(cur_query_id, [])
                        cur_sentence = self.detokenizer(cur_tokens)
                        cur_centrality = 0.0
                        if centrality_loader is not None:
                            cur_centrality = centrality_loader.get_imp_raw(cur_node_id)

                        # 區間內 Top-3（依中心性排序）
                        interval_candidates = []
                        for t in range(interval_start, interval_end + 1):
                            if t >= len(episode_examples):
                                continue
                            ex = episode_examples[t]
                            qid = ex.get("query_id", "")
                            nid = int(node_ids_batch[b][t])

                            tid = -1
                            if centrality_loader is not None and nid != -1 and hasattr(centrality_loader, "get_turn_id"):
                                try:
                                    tid = int(centrality_loader.get_turn_id(nid))
                                except Exception:
                                    tid = -1
                            if tid == -1:
                                tid = t

                            toks = dataset.query.get(qid, [])
                            sent = self.detokenizer(toks)
                            cent = 0.0
                            if centrality_loader is not None:
                                cent = centrality_loader.get_imp_raw(nid)

                            interval_candidates.append({
                                "node_id": nid,
                                "query_id": qid,
                                "turn_id": tid,
                                "sentence": sent,
                                "centrality": cent
                            })
                        interval_candidates.sort(key=lambda x: x["centrality"], reverse=True)

                        interval_start_turn_id = -1
                        interval_end_turn_id = -1
                        if interval_candidates:
                            # 由於 interval_start/interval_end 是 episode 內 index，這裡改用對應句子的 turn_id（CSV 對照）做邊界輸出
                            # 注意：區間規則固定為「包含本次 shift，不包含前一次 shift」
                            start_node_id = int(node_ids_batch[b][interval_start]) if interval_start < len(node_ids_batch[b]) else -1
                            end_node_id = cur_node_id
                            if centrality_loader is not None and hasattr(centrality_loader, "get_turn_id"):
                                try:
                                    interval_start_turn_id = int(centrality_loader.get_turn_id(start_node_id)) if start_node_id != -1 else interval_start
                                except Exception:
                                    interval_start_turn_id = interval_start
                                try:
                                    interval_end_turn_id = int(centrality_loader.get_turn_id(end_node_id)) if end_node_id != -1 else interval_end
                                except Exception:
                                    interval_end_turn_id = interval_end
                            else:
                                interval_start_turn_id = interval_start
                                interval_end_turn_id = interval_end

                        shift_events.append({
                            "shift_query_id": cur_query_id,
                            "shift_turn_id": cur_turn_id,
                            "shift_node_id": cur_node_id,
                            "shift_sentence": cur_sentence,
                            "shift_centrality": cur_centrality,
                            "interval_start_turn": interval_start,
                            "interval_end_turn": interval_end,
                            "interval_start_turn_id": interval_start_turn_id,
                            "interval_end_turn_id": interval_end_turn_id,
                            "interval_top3": interval_candidates[:3]
                        })

                        prev_shift_t = cur_t
                    shift_top3_records.append({
                        "dialog_id": dialog_id,
                        "shift_found": shift_found,
                        # 保持欄位相容：仍提供 shift_top3，但改為「第一次 shift 事件」的區間 Top-3
                        "shift_top3": (shift_events[0]["interval_top3"] if shift_events else []),
                        # 新增：完整 shift 事件清單（建議消費此欄位）
                        "shift_events": shift_events
                    })

                episode_offset += batch_size


            accumulative_final_ks_pred = torch.cat(accumulative_final_ks_pred, dim=0)
            accumulative_shift_ks_pred = torch.cat(accumulative_shift_ks_pred, dim=0)
            accumulative_inherit_ks_pred = torch.cat(accumulative_inherit_ks_pred, dim=0)

            accumulative_ID_pred = torch.cat(accumulative_ID_pred, dim=0)
            accumulative_knowledge_label = torch.cat(accumulative_knowledge_label, dim=0)
            accumulative_ID_label = torch.cat(accumulative_ID_label, dim=0)
            accumulative_episode_mask = torch.cat(accumulative_episode_mask, dim=0)
            accumulative_final_ks_acc = accuracy_score(accumulative_knowledge_label.detach().cpu(), accumulative_final_ks_pred.detach().cpu(), sample_weight=accumulative_episode_mask.detach().cpu())
            accumulative_shift_ks_acc = accuracy_score(accumulative_knowledge_label.detach().cpu(), accumulative_shift_ks_pred.detach().cpu(), sample_weight=accumulative_episode_mask.detach().cpu())
            accumulative_inherit_ks_acc = accuracy_score(accumulative_knowledge_label.detach().cpu(), accumulative_inherit_ks_pred.detach().cpu(), sample_weight=accumulative_episode_mask.detach().cpu())

            with open(os.path.join(output_path + '/ks_pred/', epoch + "_ks_pred.json"), 'w', encoding='utf-8') as w:
                json.dump(accumulative_final_ks_pred.tolist(), w)

            accumulative_ID_acc = accuracy_score(accumulative_ID_label.detach().cpu(), accumulative_ID_pred.detach().cpu(), sample_weight=accumulative_episode_mask.detach().cpu())

            valid_mask = (accumulative_episode_mask == 1) & (accumulative_ID_label != -1)
            valid_true = accumulative_ID_label[valid_mask].detach().cpu().numpy()
            valid_pred = accumulative_ID_pred[valid_mask].detach().cpu().numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(
                valid_true, valid_pred, average='binary', zero_division=0
            )
            shift_prf = {
                "precision": rounder(precision * 100, 2),
                "recall": rounder(recall * 100, 2),
                "f1": rounder(f1 * 100, 2)
            }


        return output_path, dataset.answer_file, {"ks_acc": rounder(100*(accumulative_final_ks_acc), 2)}, {"shift_ks_acc": rounder(100*(accumulative_shift_ks_acc), 2)}, {"inherit_ks_acc": rounder(100*(accumulative_inherit_ks_acc), 2)}, {"ID_acc": rounder(100*(accumulative_ID_acc), 2)}, shift_prf, shift_top3_records, shift_pred_records


    def test(self, method, dataset, collate_fn, batch_size, dataset_name, epoch, output_path):
        #  disables tracking of gradients in autograd.
        # In this mode, the result of every computation will have requires_grad=False, even when the inputs have requires_grad=True.
        with torch.no_grad():
            run_file, answer_file, final_ks_acc, shift_ks_acc, inherit_ks_acc, ID_acc, shift_prf, shift_top3_records, shift_pred_records = self.predict(
                method, dataset, collate_fn, batch_size, dataset_name+"_"+epoch, output_path
            )

        print("Start auotimatic evaluation")

        print("KNOW_ACC", final_ks_acc)
        print("shift KNOW_ACC", shift_ks_acc)
        print("inherit KNOW_ACC", inherit_ks_acc)
        print("ID_ACC", ID_acc)

        metric_output = {**final_ks_acc, **shift_ks_acc, **inherit_ks_acc, **ID_acc, **shift_prf}
        print({epoch+"_"+dataset_name: metric_output})

        try:
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'r', encoding='utf-8') as r:
                result_log = json.load(r)
            result_log[epoch + "_" + dataset_name] = metric_output
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'w', encoding='utf-8') as w:
                json.dump(result_log, w)

        except FileNotFoundError:
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'w', encoding='utf-8') as w:
                result_log={}
                result_log[epoch+"_"+dataset_name] = metric_output
                json.dump(result_log, w)

        metrics_dir = os.path.join(output_path, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        shift_metrics_path = os.path.join(metrics_dir, "shift_metrics.json")
        shift_metrics_entry = {
            "run": self.name,
            "dataset": dataset_name,
            "epoch": epoch,
            **shift_prf
        }
        try:
            with open(shift_metrics_path, 'r', encoding='utf-8') as r:
                shift_log = json.load(r)
        except FileNotFoundError:
            shift_log = {}
        shift_log[epoch + "_" + dataset_name] = shift_metrics_entry
        with open(shift_metrics_path, 'w', encoding='utf-8') as w:
            json.dump(shift_log, w, ensure_ascii=False)

        shift_top3_path = os.path.join(metrics_dir, "shift_top3.jsonl")
        with open(shift_top3_path, 'a', encoding='utf-8') as w:
            for record in shift_top3_records:
                record_out = {"run": self.name, "dataset": dataset_name, "epoch": epoch, **record}
                w.write(json.dumps(record_out, ensure_ascii=False) + "\n")

        # 6.3：逐句 shift 預測標籤（0/1）
        shift_pred_path = os.path.join(metrics_dir, "shift_pred.jsonl")
        with open(shift_pred_path, 'a', encoding='utf-8') as w:
            for record in shift_pred_records:
                record_out = {"run": self.name, "dataset": dataset_name, "epoch": epoch, **record}
                w.write(json.dumps(record_out, ensure_ascii=False) + "\n")

        ablation_csv = os.path.join(metrics_dir, "ablation_results.csv")
        if not os.path.exists(ablation_csv):
            with open(ablation_csv, 'w', encoding='utf-8') as w:
                w.write("run,dataset,epoch,precision,recall,f1\n")
        with open(ablation_csv, 'a', encoding='utf-8') as w:
            w.write("{},{},{},{},{},{}\n".format(
                self.name,
                dataset_name,
                epoch,
                shift_prf["precision"],
                shift_prf["recall"],
                shift_prf["f1"]
            ))

        return None
